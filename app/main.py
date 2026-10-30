import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.models import model_health
from backend.config import get_settings

logger = logging.getLogger('citeguard.startup')


@asynccontextmanager
async def lifespan(_app):
    task = None
    if get_settings().citeguard_mode == 'neural' and get_settings().citeguard_warmup:
        async def warm():
            health = await asyncio.to_thread(model_health)
            for name, st in health.items():
                if st['status'] == 'READY':
                    logger.info('model %s READY on %s (load %.1fs, probe %.1fs)', name, st.get('device'), st.get('load_seconds') or 0, st.get('probe_seconds') or 0)
                else:
                    logger.error('model %s UNAVAILABLE: %s at %s: %s | %s', name, st.get('cause'), st.get('stage'), st.get('error'), st.get('remedy'))
        task = asyncio.create_task(warm())  # non-blocking: the API is up while models load
    yield
    if task and not task.done():
        task.cancel()

app = FastAPI(
    title="CiteGuard API",
    description="AI-Powered Citation Verification & Evidence Alignment Platform",
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include main API routes
app.include_router(router, prefix="/api")

@app.get("/api")
def root():
    return {
        "service": "CiteGuard API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "verify_pdf": "POST /api/verify",
            "analyze_text": "POST /api/analyze-text",
            "benchmark": "GET /api/benchmark",
            "results": "GET /api/results/{job_id}",
            "export": "GET /api/export/{job_id}?format=json|csv"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "CiteGuard Backend"}


frontend = Path(__file__).resolve().parents[1] / 'citeguard-frontend' / 'dist'
if frontend.is_dir():
    app.mount('/', StaticFiles(directory=frontend, html=True), name='frontend')
