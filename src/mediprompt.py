import re
from typing import Dict, Any

# Registry for task functions
task_registry = {}

def register_task(name):
    def decorator(func):
        task_registry[name] = func
        return func
    return decorator

class MediPrompt:
    def __init__(self):
        self.context = {}

    def parse_and_execute(self, script: str):
        lines = script.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("prompt"):
                self.handle_prompt(line)
            elif line.startswith("task"):
                self.handle_task(line)
            elif line.startswith("action"):
                self.handle_action(line)
            elif line.startswith("for"):
                self.handle_loop(line)

    def handle_prompt(self, line: str):
        match = re.match(r"prompt\s+(\w+)\(([^)]*)\)", line)
        if match:
            name, params = match.groups()
            self.context[name] = {p.strip(): None for p in params.split(",")}
            print(f"[INFO] Defined prompt: {name} with params {params}")

    def handle_task(self, line: str):
        match = re.match(r"task\s+(\w+)::'([^']*)'\(\)", line)
        if match:
            task_type, task_name = match.groups()
            if task_name in task_registry:
                result = task_registry[task_name]()
                self.context[task_type] = result
                print(f"[INFO] Executed task {task_name}, result: {result}")

    def handle_action(self, line: str):
        match = re.match(r"action\s+(\w+)\(([^)]*)\)", line)
        if match:
            action_name, params = match.groups()
            print(f"[INFO] Executing action: {action_name} with params {params}")
            model = self.load_model('llama-7b')
            self.run_model(model, params)

    def handle_loop(self, line: str):
        match = re.match(r"for\s*:\s*(\w+)\s*in\s*\[(.*?)\]:\s*(\w+)\((\w+)\)", line)
        if match:
            var, cases, func, param = match.groups()
            case_ids = cases.split(',')
            for case_id in case_ids:
                print(f"[INFO] Processing case {case_id}: Executing {func}({param})")
                self.handle_action(f"action {func}({case_id})")

    def load_model(self, model_name: str):
        print(f"[INFO] Loading model {model_name}...")
        return {"model_name": model_name}

    def run_model(self, model, case_id: str):
        print(f"[INFO] Running {model['model_name']} on case {case_id}...")
        return {"prediction": "positive"}

# Example task implementations
@register_task("lesion")
def segment_lesion() -> Dict[str, Any]:
    print("[INFO] Performing lesion segmentation...")
    return {"segmentation": "lesion_mask.png"}

@register_task("cancer")
def diagnose_cancer() -> Dict[str, Any]:
    print("[INFO] Performing cancer diagnosis...")
    return {"diagnosis": "malignant"}
