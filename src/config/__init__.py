class Config:
    def __init__(self, feedback_file: str, min_samples_for_update: int):
        self.feedback = FeedbackConfig(feedback_file, min_samples_for_update)

class FeedbackConfig:
    def __init__(self, feedback_file: str, min_samples_for_update: int):
        self.feedback_file = feedback_file
        self.min_samples_for_update = min_samples_for_update