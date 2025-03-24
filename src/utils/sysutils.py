import matplotlib.pyplot as plt
import numpy as np
import os

import transformers

import models.qwen

def instantiate_model(model_name):
    if model_name == "qwen":
        pass
    else:
        return None

def exec_(cmd, args, data):
    if cmd == 'print':
        print(args)
        return
    if cmd == 'run':
        model = args[0]
        image = args[1]
        prompt = args[2]

        

        print(f"Running model {model} on image {image} with prompt {prompt}")
        return