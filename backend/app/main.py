from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PredictRequest
from app.model import predict_text

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
async def predict(request: PredictRequest):
    #run prediction model here
    return predict_text(request.text)
