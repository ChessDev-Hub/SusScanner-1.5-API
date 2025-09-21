# api/main.py
import os, importlib
from typing import Any, Dict, Optional
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── CORS ──────────────────────────────────────────────────────────────────────
origins_env = os.getenv("CORS_ORIGINS", "")
DEFAULT_ORIGINS = [
    "https://chessdev-hub.github.io",           # GitHub Pages root
    "https://chessdev-hub.github.io/SusScanner1.50",  # your site path
    "http://localhost:5173", "http://127.0.0.1:5173", # local dev
]
allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()] or DEFAULT_ORIGINS

# Optional path prefix (e.g. "/api")
API_PREFIX = os.getenv("API_PREFIX", "").strip()  # "" or "/api"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sus Scanner API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

@router.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}

# ── Lazy load scanner (supports several module locations) ─────────────────────
_ScannerCls: Optional[type] = None
_analyze_fn = None
_loaded = False

def _import_first(*names: str):
    last_err: Optional[Exception] = None
    for n in names:
        try:
            return importlib.import_module(n)
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    raise ImportError("no module names provided")

def _load_scanner_once() -> None:
    global _ScannerCls, _analyze_fn, _loaded
    if _loaded:
        return
    mod = _import_first("scanner", "services.scanner", "api.scanner")
    _ScannerCls = getattr(mod, "SusScanner", None)
    _analyze_fn = getattr(mod, "analyze_user", None) or getattr(mod, "analyze_player", None)
    _loaded = True

class ScanRequest(BaseModel):
    username: str

def _to_json(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {"result": str(obj)}

@router.post("/scan")
def scan(req: ScanRequest) -> Dict[str, Any]:
    username = (req.username or "").strip()
    if not username:
        raise HTTPException(400, "username is required")

    try:
        _load_scanner_once()
    except Exception as e:
        raise HTTPException(500, f"failed to import scanner module: {e}")

    # function-style
    if _analyze_fn is not None:
        try:
            metrics = _analyze_fn(username)
        except Exception as e:
            raise HTTPException(500, f"analyze function failed: {e}")
        return _to_json(metrics)

    # class-style
    if _ScannerCls is not None:
        try:
            scanner = _ScannerCls(lookback_months=3)
            if hasattr(scanner, "analyze_user"):
                metrics = scanner.analyze_user(username)
            elif hasattr(scanner, "analyze_player"):
                metrics = scanner.analyze_player(username)
            else:
                raise HTTPException(500, "SusScanner lacks analyze_user/analyze_player")
        except Exception as e:
            raise HTTPException(500, f"scanner failed: {e}")
        return _to_json(metrics)

    raise HTTPException(
        500,
        "scanner.py loaded but no entrypoints found. "
        "Define class SusScanner (with analyze_user/analyze_player) or export "
        "analyze_user(name)/analyze_player(name)."
    )

# Mount routes (respects API_PREFIX env var)
app.include_router(router, prefix=API_PREFIX)
