from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PredictRequest
from app.final_model import predict_text
from app.db import log_inference, get_metrics_summary
import time


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#RESTAPI endpoints

#root endpoint
@app.get("/")
async def root():
    return {"message": "API running"}

#health endpoint
@app.get("/health")
async def health():
    return {"message": "ok"}

#predictions/model endpoint
@app.post("/predict")
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    #start timer
    start = time.perf_counter()
    #run prediction model here
    response = predict_text(request.text)
    #end timer
    end = time.perf_counter()
    latency_ms = (end - start) * 1000

    background_tasks.add_task(
        log_inference,
        request.text,
        response,
        latency_ms
    )
    return response

# Get metrics summary
@app.get("/metrics")
async def metrics():
    return get_metrics_summary()
