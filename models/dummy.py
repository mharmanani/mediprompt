import torch
import numpy as np

class Dummy:
    def __init__(self):
        self.model = lambda: None

    def run(self, image, prompt):
        print("Dummy model is running")

        # Preparation for inference
        print(f"Prompt={prompt}")

        print(f"Image:{image}")

        return "Done!"