"""FastAPI application.

    uvicorn bankq.api.app:app --port 8077
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..agents.orchestrator import ResolutionOrchestrator
from ..config import ROOT, get_settings
from ..db.session import init_db
from ..logging_setup import setup_logging
from ..ml.predictor import Predictor
from ..routing.router import Router
from ..routing.rules import RuleSet
from .observability import RequestContextMiddleware
from .routes import classify, resolve, system, tickets

log = logging.getLogger("bankq.api")
WEB_DIR = ROOT / "web"


def build_router() -> Router:
    return Router(Predictor(), RuleSet.from_yaml(get_settings().rules_path))


def build_orchestrator(router: Router) -> ResolutionOrchestrator:
    s = get_settings()
    return ResolutionOrchestrator.from_paths(router, s.knowledge_dir, s.agent_policy_path,
                                             tool_latency_ms=s.tool_latency_ms, tool_workers=s.tool_workers)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    setup_logging(s.log_level, s.log_json)
    init_db()
    try:
        app.state.router = build_router()
        app.state.orchestrator = build_orchestrator(app.state.router)
        log.info("model loaded", extra={"model_version": app.state.router.model_version})
    except FileNotFoundError as e:
        app.state.router = None
        app.state.orchestrator = None
        log.error("no model available; /ready will report not_ready", extra={"error": str(e)})
    if not s.api_key_set:
        log.warning("API key auth is disabled (BANKQ_API_KEYS is empty). Do not run like this in production.")
    yield
    if getattr(app.state, "orchestrator", None) is not None:
        app.state.orchestrator.close()


app = FastAPI(
    title="Banking Query Router",
    version=__version__,
    description="Classifies customer banking messages, masks PII, applies business rules, routes them to "
                "specialised AI agents that call banking tools and retrieve policy to resolve them, and escalates "
                "risky cases to a human.",
    lifespan=lifespan,
)
app.add_middleware(RequestContextMiddleware)
app.include_router(system.router)
app.include_router(classify.router)
app.include_router(tickets.router)
app.include_router(resolve.router)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error", extra={"path": request.url.path})
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Quote the X-Request-ID when reporting."})


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/assistant", include_in_schema=False)
def assistant():
    return FileResponse(WEB_DIR / "assistant.html")
