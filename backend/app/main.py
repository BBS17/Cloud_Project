from fastapi import FastAPI
from app.schemas import PredictRequest
from app.model import predict_text

app = FastAPI()

#RESTAPI endpoints

#root endpoint
@app.get("/")
async def root():
    return {"message: API ready"}

#health endpoint
@app.get("/health")
async def root():
    return {"message": "ok"}

#predictions/model endpoint
@app.post("/predict")
async def predict(request: PredictRequest):
    #run prediction model here
    return predict_text(request.text)
