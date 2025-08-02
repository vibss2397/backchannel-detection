from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from src.models import KeyWordBaselineModel, BackChannelDetectionModel

# Define the input and output data models
class InputData(BaseModel):
    agent_text: str
    partial_transcript: str

class OutputData(BaseModel):
    is_backchannel: bool
    confidence: float


# Initialize FastAPI
app = FastAPI()
baseline_model = None
trained_model = None


# Initialize the baseline model on app startup
@app.on_event("startup")
def startup_event():
    global baseline_model, trained_model
    print("Initializing baseline model...")
    baseline_model = KeyWordBaselineModel()
    
    print("Initializing trained model...")
    trained_model = BackChannelDetectionModel()
    


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/baseline")
def run_inference_on_baseline(input: InputData) -> OutputData:
    """
    Endpoint to run inference on the baseline model.
    """
    if not baseline_model:
        return {"error": "Model not initialized. Please try again later."}
    result = baseline_model.predict(input.agent_text, input.partial_transcript)
    return OutputData(
        is_backchannel=result["is_backchannel"],
        confidence=result["confidence"]
    )


@app.post("/backchannelmodel")
def run_inference_on_trained_model(input: InputData) -> OutputData:
    """
    Endpoint to run inference on the trained model.
    """
    if not trained_model:
        return {"error": "Trained model not initialized. Please try again later."}
    
    # Assuming trained_model has a predict method similar to baseline_model
    result = trained_model.predict(input.agent_text, input.partial_transcript)
    
    return OutputData(
        is_backchannel=result["is_backchannel"],
        confidence=result["confidence"]
    )