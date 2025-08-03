import re
import joblib
import pandas as pd
import os
import logging

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Configuration ---
def get_model_paths():
    """Returns a dictionary of model paths."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        'tfidf_model': os.path.join(current_dir, 'weights/tfidf_model.joblib'),
        'fasttext_pipeline': os.path.join(current_dir, 'weights/fasttext_model.joblib'),
        'fasttext_model': os.path.join(current_dir, 'weights/cc.en.300.bin'),
    }

class KeyWordBaselineModel:
    """A baseline model that uses a keyword list to detect backchannels."""
    def __init__(self):
        self.backchannel_keywords = {
            "yeah", "yes", "uh-huh", "mhmm", "mm-hmm", "hmm",
            "oh", "ah", "uhhuh", "uh", "um", "mmmm", "yep",
            "wow", "right", "okay", "ok", "sure", "alright",
            "gotcha", "mmhmm", "great", "sweet", "ma'am", "awesome",
            "i see", "got it", "that makes sense", "i hear you",
            "i understand", "good afternoon", "hey there", "perfect",
            "that's true", "good point", "exactly", "makes sense",
            "no problem", "indeed", "certainly", "very well", "absolutely",
            "correct", "of course", "k", "hey", "hello", "hi", "yo",
            "good morning"
        }
    
    def predict(self, agent_text: str, partial_transcript: str) -> dict:
        """
        Predicts if the partial transcript is a backchannel based on keywords.
        
        Args:
            agent_text (str): The text spoken by the agent (context).
            partial_transcript (str): The current utterance to classify.
        
        Returns:
            dict: A dictionary with 'is_backchannel' and 'confidence' keys.
        """
        text = partial_transcript.lower()
        words = set(re.findall(r'\b\w+\b', text))
        
        is_backchannel = any(word in words for word in self.backchannel_keywords)
        confidence = 1.0 if is_backchannel else 0.0
        
        return {
            "is_backchannel": is_backchannel,
            "confidence": confidence
        }


class BackChannelDetectionModel:
    """A model that uses a trained TF-IDF pipeline to detect backchannels."""
    def __init__(self):
        model_paths = get_model_paths()
        self.weights = model_paths['tfidf_model']
        
        if not os.path.exists(self.weights):
            logger.error(f"TF-IDF model file not found at {self.weights}")
            raise FileNotFoundError(f"TF-IDF model file not found at {self.weights}")
            
        try:
            self.pipeline = joblib.load(self.weights)
            logger.info("TF-IDF model loaded successfully.")
        except Exception as e:
            logger.exception(f"Failed to load TF-IDF model: {e}")
            raise
    
    def predict(self, agent_text: str, partial_transcript: str) -> dict:
        """
        Predicts if the partial transcript is a backchannel using the TF-IDF model.
        
        Args:
            agent_text (str): The text spoken by the agent (context).
            partial_transcript (str): The current utterance to classify.
        
        Returns:
            dict: A dictionary with 'is_backchannel' and 'confidence' keys.
        """
        input_df = pd.DataFrame([{
            'previous_utter_clean': agent_text,
            'current_utter_clean': partial_transcript
        }])
        
        try:
            prediction = self.pipeline.predict(input_df)
            confidence = self.pipeline.predict_proba(input_df)[0][1]
            return {
                "is_backchannel": bool(prediction[0]),
                "confidence": confidence
            }
        except Exception as e:
            logger.exception(f"An error occurred during TF-IDF prediction: {e}")
            raise

class FastTextBackChannelModel:
    """A model that uses a trained FastText pipeline to detect backchannels."""
    def __init__(self):
        model_paths = get_model_paths()
        pipeline_path = model_paths['fasttext_pipeline']
        fasttext_model_path = model_paths['fasttext_model']

        if not os.path.exists(pipeline_path) or not os.path.exists(fasttext_model_path):
            logger.error(f"FastText model files not found. Searched for {pipeline_path} and {fasttext_model_path}")
            raise FileNotFoundError("FastText model files not found.")

        try:
            self.pipeline = joblib.load(pipeline_path)
            self.pipeline.named_steps['vectorizer'].model_path = fasttext_model_path
            logger.info("FastText model loaded and configured successfully.")
        except Exception as e:
            logger.exception(f"Failed to load or configure FastText model: {e}")
            raise
    
    def predict(self, agent_text: str, partial_transcript: str) -> dict:
        """
        Predicts if the partial transcript is a backchannel using the FastText model.
        
        Args:
            agent_text (str): The text spoken by the agent (context).
            partial_transcript (str): The current utterance to classify.
        
        Returns:
            dict: A dictionary with 'is_backchannel' and 'confidence' keys.
        """
        input_df = pd.DataFrame([{
            'previous_utter_clean': agent_text,
            'current_utter_clean': partial_transcript
        }])
        
        try:
            prediction = self.pipeline.predict(input_df)
            confidence = self.pipeline.predict_proba(input_df)[0][1]
            return {
                "is_backchannel": bool(prediction[0]),
                "confidence": confidence
            }
        except Exception as e:
            logger.exception(f"An error occurred during FastText prediction: {e}")
            raise
