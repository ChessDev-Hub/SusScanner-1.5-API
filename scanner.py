#!/usr/bin/env python3
from __future__ import annotations
import csv
import datetime as dt
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import requests
import os


# ---------------- models ----------------


@dataclass
class GameSummary:
    result: str
    is_win: bool
    is_loss: bool
    is_draw: bool
    my_rating: Optional[int]
    opp_rating: Optional[int]
    plies: Optional[int]
    end_reason: str
    is_rated: bool
    from_tournament: bool
    end_time: Optional[int]


@dataclass
class UserMetrics:
    username: str

    # lifetime (in fetched window; rated daily)
    lifetime_games: int = 0
    lifetime_wins: int = 0
    lifetime_draws: int = 0
    lifetime_losses: int = 0

    # recent (same window)
    recent_games: int = 0
    recent_wins: int = 0
    recent_draws: int = 0
    recent_losses: int = 0

    # streaks (all daily)
    win_streak: int = 0
    max_win_streak: int = 0

    # context signals
    upset_wins: int = 0
    short_win_rate: float = 0.0
    timeout_win_ratio: float = 0.0  # context only (wins by opp resign/timeout)

    # tourney vs non-tourney splits (rated daily)
    tourn_games: int = 0
    tourn_wins: int = 0
    tourn_draws: int = 0
    tourn_losses: int = 0
    non_tourn_games: int = 0
    non_tourn_wins: int = 0
    non_tourn_draws: int = 0
    non_tourn_losses: int = 0
    tourn_win_rate: float = 0.0
    non_tourn_win_rate: float = 0.0
    wr_gap: float = 0.0  # legacy display only

    # Elo accounting
    elo_gain: float = 0.0  # sum(max(0, actual - expected))
    elo_loss: float = 0.0  # sum(max(0, expected - actual))
    elo_ratio: float = 0.0  # gain / max(loss, eps)

    tourn_elo_gain: float = 0.0
    tourn_elo_loss: float = 0.0
    tourn_elo_ratio: float = 0.0

    non_tourn_elo_gain: float = 0.0
    non_tourn_elo_loss: float = 0.0
    non_tourn_elo_ratio: float = 0.0

    elo_ratio_gap: float = 0.0  # tournament − non-tournament

    # sandbagging (player's OWN losses via resign/timeout/abandon)
    t_self_bail_loss_ratio: float = 0.0
    nt_self_bail_loss_ratio: float = 0.0

    suspicion_score: float = 0.0
    reasons: List[str] = field(default_factory=list)


# ---------------- scanner ----------------


class SusScanner:
    API_BASE = "https://api.chess.com/pub/player"

    def __init__(
        self,
        *,
        lookback_months: int = 2,
        min_lifetime_games: int = 30,
        # context only
        rating_upset_margin: int = 250,
        short_game_plies: int = 40,
        short_win_rate: float = 0.70,
        streak_suspect: int = 8,
        tourn_min_games: int = 15,
        non_tourn_min_games: int = 15,
        request_timeout: int = 20,
        # Elo thresholds
        min_games_for_elo: int = 20,
        high_elo_ratio: float = 2.0,
        elo_ratio_gap_suspect: float = 1.0,
        # sandbagging thresholds
        min_nt_losses_for_flag: int = 6,
        high_bail_loss_ratio: float = 0.60,
        gap_vs_tournament: float = 0.20,
        user_agent: Optional[str] = None,
    ):
        self.lookback_months = lookback_months
        self.min_lifetime_games = min_lifetime_games
        self.rating_upset_margin = rating_upset_margin
        self.short_game_plies = short_game_plies
        self.short_win_rate_th = short_win_rate
        self.streak_suspect = streak_suspect
        self.tourn_min_games = tourn_min_games
        self.non_tourn_min_games = non_tourn_min_games

        self.min_games_for_elo = min_games_for_elo
        self.high_elo_ratio = high_elo_ratio
        self.elo_ratio_gap_suspect = elo_ratio_gap_suspect

        self.min_nt_losses_for_flag = min_nt_losses_for_flag
        self.high_bail_loss_ratio = high_bail_loss_ratio
        self.gap_vs_tournament = gap_vs_tournament

        self.request_timeout = request_timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent or self._default_user_agent()}
        )

    # ---------- public API ----------

    def analyze_usernames_file(self, path: str) -> List[UserMetrics]:
        with open(path, "r", encoding="utf-8") as f:
            usernames = [ln.strip().lower() for ln in f if ln.strip()]
        return self.analyze_usernames(usernames)

    def analyze_usernames(self, usernames: List[str]) -> List[UserMetrics]:
        out: List[UserMetrics] = []
        for u in usernames:
            try:
                out.append(self.analyze_user(u))
            except Exception as e:
                print(f"[warn] {u}: {e}")
        out.sort(key=lambda m: m.suspicion_score, reverse=True)
        return out

    def analyze_user(self, username: str) -> UserMetrics:
        games = self._fetch_daily_games(username, self.lookback_months)
        m = UserMetrics(username=username)
        if not games:
            return m

        rated = [g for g in games if g.is_rated]

        # lifetime (window)
        m.lifetime_games = len(rated)
        m.lifetime_wins = sum(g.is_win for g in rated)
        m.lifetime_draws = sum(g.is_draw for g in rated)
        m.lifetime_losses = sum(g.is_loss for g in rated)

        # recent == window
        m.recent_games = m.lifetime_games
        m.recent_wins = m.lifetime_wins
        m.recent_draws = m.lifetime_draws
        m.recent_losses = m.lifetime_losses

        # streaks (all daily)
        m.win_streak, m.max_win_streak = self._rolling_win_streak(games)

        # helper: bail finishes (opp resign/timeout/abandon)
        def _is_bail(reason: str) -> bool:
            r = (reason or "").lower()
            return ("resign" in r) or ("timeout" in r) or ("abandon" in r)

        # upset wins (ignore if opponent bailed)
        for g in rated:
            if g.is_win and g.my_rating and g.opp_rating:
                if (
                    g.opp_rating - g.my_rating >= self.rating_upset_margin
                    and not _is_bail(g.end_reason)
                ):
                    m.upset_wins += 1

        # short wins & timeoutish wins (context only)
        win_games = [g for g in rated if g.is_win]
        short_wins = [
            g for g in win_games if (g.plies or 10**9) <= self.short_game_plies
        ]
        m.short_win_rate = self._ratio(len(short_wins), max(1, len(win_games)))
        timeoutish = [g for g in win_games if _is_bail(g.end_reason)]
        m.timeout_win_ratio = self._ratio(len(timeoutish), max(1, len(win_games)))

        # tourney vs non-tourney sets
        t_games = [g for g in rated if g.from_tournament]
        nt_games = [g for g in rated if not g.from_tournament]

        m.tourn_games = len(t_games)
        m.tourn_wins = sum(g.is_win for g in t_games)
        m.tourn_draws = sum(g.is_draw for g in t_games)
        m.tourn_losses = sum(g.is_loss for g in t_games)

        m.non_tourn_games = len(nt_games)
        m.non_tourn_wins = sum(g.is_win for g in nt_games)
        m.non_tourn_draws = sum(g.is_draw for g in nt_games)
        m.non_tourn_losses = sum(g.is_loss for g in nt_games)

        # legacy win-rate (display only)
        m.tourn_win_rate = self._ratio(
            m.tourn_wins, max(1, m.tourn_games - m.tourn_draws)
        )
        m.non_tourn_win_rate = self._ratio(
            m.non_tourn_wins, max(1, m.non_tourn_games - m.non_tourn_draws)
        )
        m.wr_gap = m.tourn_win_rate - m.non_tourn_win_rate

        # ---- Elo accounting ----
        def expected_score(my: Optional[int], opp: Optional[int]) -> Optional[float]:
            if my is None or opp is None:
                return None
            return 1.0 / (1.0 + 10.0 ** ((opp - my) / 400.0))

        def actual_score(g: GameSummary) -> float:
            return 1.0 if g.is_win else (0.5 if g.is_draw else 0.0)

        def accum_elo(glist: List[GameSummary]) -> Tuple[float, float]:
            gain = loss = 0.0
            for g in glist:
                e = expected_score(g.my_rating, g.opp_rating)
                if e is None:
                    continue
                a = actual_score(g)
                d = a - e
                if d >= 0:
                    gain += d
                else:
                    loss += -d
            return gain, loss

        m.elo_gain, m.elo_loss = accum_elo(rated)
        eps = 1e-9
        m.elo_ratio = m.elo_gain / (m.elo_loss if m.elo_loss > eps else eps)

        m.tourn_elo_gain, m.tourn_elo_loss = accum_elo(t_games)
        m.non_tourn_elo_gain, m.non_tourn_elo_loss = accum_elo(nt_games)

        m.tourn_elo_ratio = (
            (m.tourn_elo_gain / (m.tourn_elo_loss if m.tourn_elo_loss > eps else eps))
            if m.tourn_games
            else 0.0
        )
        m.non_tourn_elo_ratio = (
            (
                m.non_tourn_elo_gain
                / (m.non_tourn_elo_loss if m.non_tourn_elo_loss > eps else eps)
            )
            if m.non_tourn_games
            else 0.0
        )
        m.elo_ratio_gap = m.tourn_elo_ratio - m.non_tourn_elo_ratio

        # ---- sandbagging: player's OWN losses via bail in NT ----
        t_losses = [g for g in t_games if g.is_loss]
        nt_losses = [g for g in nt_games if g.is_loss]
        t_bail_losses = [g for g in t_losses if _is_bail(g.end_reason)]
        nt_bail_losses = [g for g in nt_losses if _is_bail(g.end_reason)]
        m.t_self_bail_loss_ratio = self._ratio(
            len(t_bail_losses), max(1, len(t_losses))
        )
        m.nt_self_bail_loss_ratio = self._ratio(
            len(nt_bail_losses), max(1, len(nt_losses))
        )

        # ---- scoring (Elo + sandbagging) ----
        score = 0.0

        def bump(pts: float, reason: str):
            nonlocal score
            score += pts
            m.reasons.append(reason)

        if (
            m.lifetime_games >= self.min_games_for_elo
            and m.elo_ratio >= self.high_elo_ratio
        ):
            bump(
                2.0,
                f"High EloRatio {m.elo_ratio:.2f} over {m.lifetime_games} rated games",
            )

        if (
            m.tourn_games >= self.tourn_min_games
            and m.non_tourn_games >= self.non_tourn_min_games
            and m.elo_ratio_gap >= self.elo_ratio_gap_suspect
        ):
            bump(
                2.2,
                (
                    f"Tourn EloRatio {m.tourn_elo_ratio:.2f} vs non-tourn "
                    f"{m.non_tourn_elo_ratio:.2f} (gap {m.elo_ratio_gap:.2f})"
                ),
            )

        # sandbagging bump
        if (
            len(nt_losses) >= self.min_nt_losses_for_flag
            and m.nt_self_bail_loss_ratio >= self.high_bail_loss_ratio
        ):
            if (
                m.nt_self_bail_loss_ratio - m.t_self_bail_loss_ratio
            ) >= self.gap_vs_tournament or len(t_losses) < 3:
                bump(
                    1.5,
                    (
                        "High non-tournament self-bail losses "
                        f"{m.nt_self_bail_loss_ratio:.0%}"
                    ),
                )

        # orthogonal context
        if m.win_streak >= self.streak_suspect:
            bump(1.0, f"Active win streak {m.win_streak}")
        if m.upset_wins >= 3:
            bump(1.0, f"{m.upset_wins} upset wins (>={self.rating_upset_margin})")
        if m.short_win_rate >= self.short_win_rate_th and len(win_games) >= 10:
            bump(
                0.7, f"{m.short_win_rate:.0%} of wins <= {self.short_game_plies} plies"
            )
        # NOTE: no bump for timeout_win_ratio (context only)

        m.suspicion_score = round(score, 2)
        return m

    # ---------- output ----------

    def print_table(self, metrics: List[UserMetrics]) -> None:
        header = (
            f"{'User':<20} {'Games':>5} {'W-D-L':>9} "
            f"{'EloR':>6} {'T.EloR':>7} {'NT.EloR':>7} {'ΔEloR':>7} "
            f"{'Stk':>4} {'Upset':>5} {'ShortW%':>8} {'TO/Res%':>8} "
            f"{'NT-SelfTO%':>10} {'T-SelfTO%':>9} "
            f"{'TWR':>6} {'NTWR':>6} {'ΔWR':>6} {'Score':>6}"
        )
        print(header)
        print("-" * len(header))
        for m in metrics:
            wdl = f"{m.lifetime_wins}-{m.lifetime_draws}-{m.lifetime_losses}"
            print(
                f"{m.username:<20} {m.lifetime_games:>5} {wdl:>9} "
                f"{m.elo_ratio:>6.2f} {m.tourn_elo_ratio:>7.2f} "
                f"{m.non_tourn_elo_ratio:>7.2f} {m.elo_ratio_gap:>7.2f} "
                f"{m.win_streak:>4} {m.upset_wins:>5} {m.short_win_rate:>8.0%} "
                f"{m.timeout_win_ratio:>8.0%} {m.nt_self_bail_loss_ratio:>10.0%} "
                f"{m.t_self_bail_loss_ratio:>9.0%} {m.tourn_win_rate:>6.0%} "
                f"{m.non_tourn_win_rate:>6.0%} {m.wr_gap:>6.0%} "
                f"{m.suspicion_score:>6.2f}"
            )

        print("\nTop reasons per user:")
        for m in metrics:
            if m.reasons:
                print(f"- {m.username}: {', '.join(m.reasons)}")

    @staticmethod
    def _sanitize_reasons(text: str) -> str:
        if not text:
            return text
        return text.replace("≥", ">=").replace("≤", "<=").replace("Δ", "Delta")

    def write_csv(self, path: str, metrics: List[UserMetrics]) -> None:
        # Excel-safe: utf-8-sig adds BOM so Windows Excel auto-detects UTF-8
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "username",
                    "lifetime_games",
                    "lifetime_wins",
                    "lifetime_draws",
                    "lifetime_losses",
                    "elo_gain",
                    "elo_loss",
                    "elo_ratio",
                    "tourn_games",
                    "tourn_wins",
                    "tourn_draws",
                    "tourn_losses",
                    "tourn_elo_gain",
                    "tourn_elo_loss",
                    "tourn_elo_ratio",
                    "non_tourn_games",
                    "non_tourn_wins",
                    "non_tourn_draws",
                    "non_tourn_losses",
                    "non_tourn_elo_gain",
                    "non_tourn_elo_loss",
                    "non_tourn_elo_ratio",
                    "elo_ratio_gap",
                    "short_win_rate",
                    "timeout_win_ratio",
                    "tourn_win_rate",
                    "non_tourn_win_rate",
                    "wr_gap",
                    "nt_self_bail_loss_ratio",
                    "t_self_bail_loss_ratio",
                    "suspicion_score",
                    "reasons",
                ]
            )
            for m in metrics:
                reasons_txt = self._sanitize_reasons(" | ".join(m.reasons))
                w.writerow(
                    [
                        m.username,
                        m.lifetime_games,
                        m.lifetime_wins,
                        m.lifetime_draws,
                        m.lifetime_losses,
                        f"{m.elo_gain:.6f}",
                        f"{m.elo_loss:.6f}",
                        f"{m.elo_ratio:.6f}",
                        m.tourn_games,
                        m.tourn_wins,
                        m.tourn_draws,
                        m.tourn_losses,
                        f"{m.tourn_elo_gain:.6f}",
                        f"{m.tourn_elo_loss:.6f}",
                        f"{m.tourn_elo_ratio:.6f}",
                        m.non_tourn_games,
                        m.non_tourn_wins,
                        m.non_tourn_draws,
                        m.non_tourn_losses,
                        f"{m.non_tourn_elo_gain:.6f}",
                        f"{m.non_tourn_elo_loss:.6f}",
                        f"{m.non_tourn_elo_ratio:.6f}",
                        f"{m.elo_ratio_gap:.6f}",
                        f"{m.short_win_rate:.6f}",
                        f"{m.timeout_win_ratio:.6f}",
                        f"{m.tourn_win_rate:.6f}",
                        f"{m.non_tourn_win_rate:.6f}",
                        f"{m.wr_gap:.6f}",
                        f"{m.nt_self_bail_loss_ratio:.6f}",
                        f"{m.t_self_bail_loss_ratio:.6f}",
                        f"{m.suspicion_score:.2f}",
                        reasons_txt,
                    ]
                )

    # ---------- internals ----------

    def _default_user_agent(self) -> str:
        email = os.getenv("USER_EMAIL") or "example@example.com"
        return f"GrandTourneys Suspicion Scanner (+contact: {email})"

    def _get_json(self, url: str, retries: int = 3, backoff: float = 0.8):
        for i in range(retries):
            r = self.session.get(url, timeout=self.request_timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            time.sleep(backoff * (i + 1))
        return None

    def _get_archives(self, username: str) -> List[str]:
        data = self._get_json(f"{self.API_BASE}/{username}/games/archives")
        if not data or "archives" not in data:
            return []
        return data["archives"]

    @staticmethod
    def _parse_pgn_value(pgn: str, key: str) -> Optional[str]:
        m = re.search(rf'\[{re.escape(key)}\s+"([^"]+)"\]', pgn)
        return m.group(1) if m else None

    @staticmethod
    def _parse_plies_from_pgn(pgn: str) -> Optional[int]:
        parts = re.split(r"\n\n", pgn, maxsplit=1)
        if len(parts) < 2:
            return None
        body = parts[1]
        fullmoves = len(re.findall(r"\b\d+\.", body))
        return 2 * fullmoves if fullmoves > 0 else None

    def _fetch_daily_games(
        self, username: str, lookback_months: int
    ) -> List[GameSummary]:
        archives = self._get_archives(username)
        if not archives:
            return []

        ym_to_include = set()
        today = dt.date.today()
        cur = today.replace(day=1)
        for _ in range(lookback_months):
            ym_to_include.add(f"{cur.year}-{cur.month:02d}")
            cur = (cur - dt.timedelta(days=1)).replace(day=1)

        out: List[GameSummary] = []
        for url in archives:
            parts = url.rstrip("/").split("/")
            if len(parts) < 2:
                continue
            y, m = parts[-2], parts[-1]
            ym = f"{y}-{m}"
            if ym not in ym_to_include:
                continue

            data = self._get_json(url)
            if not data or "games" not in data:
                continue

            for g in data["games"]:
                if g.get("time_class") != "daily":
                    continue
                rated = bool(g.get("rated"))
                pgn = g.get("pgn", "")
                end_reason = (self._parse_pgn_value(pgn, "Termination") or "").lower()
                result = self._parse_pgn_value(pgn, "Result") or ""

                white = g.get("white", {})
                black = g.get("black", {})
                end_time = g.get("end_time")

                is_white = white.get("username", "").lower() == username.lower()
                me = white if is_white else black
                opp = black if is_white else white
                my_rating = me.get("rating")
                opp_rating = opp.get("rating")
                plies = self._parse_plies_from_pgn(pgn)
                from_tourn = "tournament" in g

                is_win = (result == "1-0" and is_white) or (
                    result == "0-1" and not is_white
                )
                is_loss = (result == "0-1" and is_white) or (
                    result == "1-0" and not is_white
                )
                is_draw = result == "1/2-1/2"

                out.append(
                    GameSummary(
                        result=result,
                        is_win=is_win,
                        is_loss=is_loss,
                        is_draw=is_draw,
                        my_rating=my_rating,
                        opp_rating=opp_rating,
                        plies=plies,
                        end_reason=end_reason,
                        is_rated=rated,
                        from_tournament=from_tourn,
                        end_time=end_time,
                    )
                )

        return out

    @staticmethod
    def _ratio(n: int, d: int) -> float:
        return n / d if d > 0 else 0.0

    @staticmethod
    def _rolling_win_streak(games: List[GameSummary]) -> Tuple[int, int]:
        gs = [g for g in games if g.end_time]
        gs.sort(key=lambda x: x.end_time)
        cur = mx = 0
        for g in gs:
            if g.is_win:
                cur += 1
                mx = max(mx, cur)
            else:
                cur = 0
        return cur, mx

    # ---------- extras ----------

    def fetch_members(self) -> List[str]:
        """
        Returns all usernames in the Grand Tourneys club (lowercased).
        """
        url = "https://api.chess.com/pub/club/grand-tourneys/members"
        data = self.session.get(url, timeout=30).json()
        usernames = [
            m["username"].lower()
            for bucket in ["weekly", "monthly", "all_time"]
            for m in data.get(bucket, [])
            if "username" in m
        ]
        print(f"Total members: {len(usernames)}")
        return usernames


# ------------- optional local run -------------
if __name__ == "__main__":
    # Example: use club members by default if usernames.txt not provided
    import argparse as _arg

    ap = _arg.ArgumentParser()
    ap.add_argument("userfile", nargs="?", help="file with usernames (one per line)")
    ap.add_argument("--months", type=int, default=3, help="lookback months")
    ap.add_argument("--csv", default=None, help="output CSV path")
    args = ap.parse_args()

    scanner = SusScanner(lookback_months=args.months)

    if args.userfile and os.path.isfile(args.userfile):
        with open(args.userfile, "r", encoding="utf-8") as f:
            usernames = [ln.strip() for ln in f if ln.strip()]
    else:
        #usernames = scanner.fetch_members()
        usernames = ["DR4GnDrop"]  # Placeholder for actual usernames

    results = scanner.analyze_usernames(usernames)
    scanner.print_table(results)
    out = (
        args.csv
        or f"suspicion_report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    scanner.write_csv(out, results)
    print(f"\nCSV written to {out}")
