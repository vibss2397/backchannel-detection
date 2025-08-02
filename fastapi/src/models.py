import re
import joblib
import pandas as pd
import os


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
            'previous_clean': agent_text,
            'current_clean': partial_transcript
        }])
        prediction = self.pipeline.predict(input)
        confidence = self.pipeline.predict_proba(input)[0][1]
        return {
            "is_backchannel": bool(prediction[0]),
            "confidence": confidence
        }