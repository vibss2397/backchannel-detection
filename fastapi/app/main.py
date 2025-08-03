from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

from src.models import KeyWordBaselineModel, BackChannelDetectionModel, FastTextBackChannelModel

# --- API Documentation ---
API_TITLE = "Backchannel Detection API"
API_DESCRIPTION = """
An API for detecting backchannels in conversations. Provides three models:
- **Baseline**: A simple keyword-based model.
- **TF-IDF**: A machine learning model using TF-IDF features.
- **FastText**: A machine learning model using FastText embeddings.
"""
API_VERSION = "1.0.0"

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Input and Output Models ---
class InputData(BaseModel):
    agent_text: str
    partial_transcript: str

    class Config:
        schema_extra = {
            "example": {
                "agent_text": "I see.",
                "partial_transcript": "I was thinking about going to the store."
            }
        }

class OutputData(BaseModel):
    is_backchannel: bool
    confidence: float

    class Config:
        schema_extra = {
            "example": {
                "is_backchannel": True,
                "confidence": 0.9
            }
        }

# --- FastAPI App Initialization ---
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# --- Global Variables ---
baseline_model = None
tfidf_model = None
fasttext_model = None

# --- Middleware for Exception Handling ---
@app.middleware("http")
async def exception_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred."},
        )

# --- Model Loading ---
@app.on_event("startup")
def startup_event():
    global baseline_model, tfidf_model, fasttext_model
    logger.info("Initializing models...")
    try:
        baseline_model = KeyWordBaselineModel()
        tfidf_model = BackChannelDetectionModel()
        fasttext_model = FastTextBackChannelModel()
        logger.info("Models initialized successfully.")
    except Exception as e:
        logger.exception(f"Failed to initialize models: {e}")
        # Depending on the desired behavior, you might want to exit the application
        # if models fail to load. For now, we'll log the error and continue.

# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    return {"message": f"Welcome to the {API_TITLE}"}

@app.post("/baseline", response_model=OutputData, tags=["Models"])
def run_inference_on_baseline(input: InputData) -> OutputData:
    """
    Endpoint to run inference on the baseline model.
    """
    if not baseline_model:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Model not initialized. Please try again later."}
        )
    
    logger.info(f"Received baseline request: {input.dict()}")
    result = baseline_model.predict(input.agent_text, input.partial_transcript)
    logger.info(f"Baseline prediction: {result}")
    
    return OutputData(
        is_backchannel=result["is_backchannel"],
        confidence=result["confidence"]
    )

@app.post("/tfidfmodel", response_model=OutputData, tags=["Models"])
def run_inference_on_trained_model(input: InputData) -> OutputData:
    """
    Endpoint to run inference on the trained TF-IDF model.
    """
    if not tfidf_model:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Trained model not initialized. Please try again later."}
        )
    
    logger.info(f"Received TF-IDF request: {input.dict()}")
    result = tfidf_model.predict(input.agent_text, input.partial_transcript)
    logger.info(f"TF-IDF prediction: {result}")
    
    return OutputData(
        is_backchannel=result["is_backchannel"],
        confidence=result["confidence"]
    )

@app.post("/fasttextmodel", response_model=OutputData, tags=["Models"])
def run_inference_on_fasttext_model(input: InputData) -> OutputData:
    """
    Endpoint to run inference on the trained FastText model.
    """
    if not fasttext_model:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Trained model not initialized. Please try again later."}
        )
    
    logger.info(f"Received FastText request: {input.dict()}")
    result = fasttext_model.predict(input.agent_text, input.partial_transcript)
    logger.info(f"FastText prediction: {result}")
    
    return OutputData(
        is_backchannel=result["is_backchannel"],
        confidence=result["confidence"]
    )
