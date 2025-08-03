# locustfile.py

from locust import HttpUser, task, between
import random
import os

# Get the target host from environment variables, with a default value
# To set the host, run `export TARGET_HOST=http://your-host.com` in your terminal
# or `TARGET_HOST=http://your-host.com locust -f stress_test/locustfile.py`
TARGET_HOST = os.environ.get("TARGET_HOST", "http://localhost:80")

class BackchannelApiUser(HttpUser):
    host = TARGET_HOST
    # Wait 0.1 to 0.5 seconds between requests
    wait_time = between(0.1, 0.5)

    # Sample utterances for testing
    sample_utterances = [
        {"agent_text": "We have a meeting scheduled for 3pm.", "partial_transcript": "okay"},
        {"agent_text": "I think that's the best course of action.", "partial_transcript": "makes sense"},
        {"agent_text": "The system should be back online shortly.", "partial_transcript": "I need to reschedule"},
        {"agent_text": "Did you get the email I sent this morning?", "partial_transcript": "yep got it"}
    ]

    @task(1) # This task will be picked 1 time out of 3
    def predict_baseline(self):
        payload = random.choice(self.sample_utterances)
        self.client.post("/baseline", json=payload)

    @task(1) # This task will be picked 1 times out of 3
    def predict_ml_model(self):
        payload = random.choice(self.sample_utterances)
        self.client.post("/tfidfmodel", json=payload)
    
    @task(1) # This task will be picked 1 times out of 3
    def predict_fasttext_model(self):
        payload = random.choice(self.sample_utterances)
        self.client.post("/fasttextmodel", json=payload)
