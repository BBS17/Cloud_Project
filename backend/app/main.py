import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool

from app.schemas import PredictRequest, PredictResponse
from app.final_model import predict_text, is_model_loaded, load_model
from app.db import log_inference, get_metrics_summary, check_db

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please wait a moment before trying again."},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Fact Checker API starting up")
    await run_in_threadpool(load_model)
    yield
    logger.info("Fact Checker API shutting down")


app = FastAPI(
    title="Fact Checker API",
    version="1.0.0",
    description="Misinformation detection API powered by DistilBERT.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

# Restrict origins via environment variable; defaults to localhost for local dev.
# In production set ALLOWED_ORIGINS to the extension's chrome-extension:// origin.
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/")
async def root():
    return {"message": "Fact Checker API running"}


@app.get("/health")
async def health():
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        await run_in_threadpool(check_db)
    except Exception:
        raise HTTPException(status_code=503, detail="Database unreachable")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
@limiter.limit("30/minute")
async def predict(request: Request, body: PredictRequest, background_tasks: BackgroundTasks):
    start = time.perf_counter()
    try:
        # Run synchronous ML inference in a thread pool to avoid blocking the event loop
        response = await run_in_threadpool(predict_text, body.text)
    except Exception as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail="Inference failed")
    latency_ms = (time.perf_counter() - start) * 1000

    background_tasks.add_task(log_inference, body.text, response, latency_ms)
    return response


@app.get("/metrics")
async def metrics():
    result = get_metrics_summary()
    if result is None:
        raise HTTPException(status_code=503, detail="Metrics temporarily unavailable")
    return result
