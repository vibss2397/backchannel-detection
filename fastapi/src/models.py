import re
import joblib
import pandas as pd
import os

# Import the classes from src.models - this MUST match the training imports
from sharedlib.transformer_utils import ImprovedFastTextVectorizer, ColumnSelector

class KeyWordBaselineModel:
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
        Only uses the partial transcript to determine a backchannel. Agent text is not needed.
        Args:
            agent_text (str): The text spoken by the agent.
            partial_transcript (str): The partial transcript of the conversation.
        
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
    def __init__(self):
        # Construct an absolute path to the weights file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights = os.path.join(current_dir, 'weights/trained_model.joblib')
        self.pipeline = joblib.load(self.weights)
    
    
    def predict(self, agent_text: str, partial_transcript: str) -> dict:
        """
        Uses the trained model to predict if the partial transcript is a backchannel.
        Args:
            agent_text (str): The text spoken by the agent.
            partial_transcript (str): The partial transcript of the conversation.
        Returns:
            dict: A dictionary with 'is_backchannel' and 'confidence' keys.
        """
        input = pd.DataFrame([{
            'previous_utter_clean': agent_text,
            'current_utter_clean': partial_transcript
        }])
        prediction = self.pipeline.predict(input)
        confidence = self.pipeline.predict_proba(input)[0][1]
        return {
            "is_backchannel": bool(prediction[0]),
            "confidence": confidence
        }


class FastTextBackChannelModel:
    def __init__(self, model_dir='weights'):
        """
        Initializes the FastText-based backchannel detection model.

        Args:
            model_dir (str): The directory containing the model files.
        """
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the scikit-learn pipeline file
        pipeline_path = os.path.join(current_script_dir, model_dir, 'fasttext_model.joblib') # Use a generic name
        
        # Path to the FastText embedding model (.bin file)
        fasttext_model_path = os.path.join(current_script_dir, model_dir, 'cc.en.300.bin')

        if not os.path.exists(pipeline_path) or not os.path.exists(fasttext_model_path):
            raise FileNotFoundError(
                f"Model files not found. Ensure '{pipeline_path}' and "
                f"'{fasttext_model_path}' exist."
            )

        print(f"Loading pipeline from {pipeline_path}...")
        self.pipeline = joblib.load(pipeline_path)

        # The pipeline was trained with a path to the .bin file, which might now be
        # different. We must update it to the correct location for inference.
        # The 'vectorizer' step name comes from the pipeline definition in train.py
        print("Updating FastText model path in the pipeline...")
        self.pipeline.named_steps['vectorizer'].model_path = fasttext_model_path
    
    
    def predict(self, agent_text: str, partial_transcript: str) -> dict:
        """
        Uses the trained FastText model to predict if the partial transcript is a backchannel.
        
        Args:
            agent_text (str): The text spoken by the agent (context).
            partial_transcript (str): The current utterance to classify.
        
        Returns:
            dict: A dictionary with 'is_backchannel' and 'confidence' keys.
        """
        # The FastText model was trained only on the current utterance, 
        # but we keep the signature the same for consistency.
        input_df = pd.DataFrame([{
            'previous_utter_clean': agent_text,
            'current_utter_clean': partial_transcript
        }])
        
        # The pipeline will automatically select 'current_utter_clean'
        prediction = self.pipeline.predict(input_df)
        confidence = self.pipeline.predict_proba(input_df)[0][1]
        
        return {
            "is_backchannel": bool(prediction[0]),
            "confidence": confidence
        }
