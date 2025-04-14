import matplotlib.pyplot as plt
import numpy as np
import os
import time

import transformers

from models.qwen import Qwen
from models.biomedclip import BiomedCLIP
from models.llava_med import LlavaMed
from models.llama import Llama
from models.mistral import Mistral
from models.dummy import Dummy

model_registry = {
    "Qwen": Qwen,
    "BiomedCLIP": BiomedCLIP,
    "LlavaMed": LlavaMed,
    "Llama": Llama,
    "Mistral": Mistral,
    "Dummy": Dummy,
}

def exec_(cmd, args, data):
    if cmd == 'print':
        for arg in args:
            print(arg)
        return
    if cmd == 'run':
        try:
            model = model_registry[args[0]]()
        except KeyError:
            raise SyntaxError(f"Invalid model: {args[0]}")
        
        image = args[1]
        prompt = args[2]
        log = args[3] if len(args) > 3 else None
        export_path = args[4] if log else None
        output = model.run(image, prompt)
        if log:
            with open(os.path.join("logs", f"{export_path}.log"), "a") as f:
                for o in output:
                    f.write(f"[{time.ctime()}] {args[0].upper()}: {o}\n")
        else:
            for o in output:
                print(o)
        return output
    if cmd == 'for':
        varname = args[0]
        min_val, max_val = args[1], args[2]
        min_val = int(min_val)
        max_val = int(max_val)
        for i in range(min_val, max_val+1):
            data[varname] = i
            print(f"{varname} = {i}")