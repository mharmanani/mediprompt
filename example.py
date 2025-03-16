import re
from typing import Dict, Any

# Registry for task functions
task_registry = {}

def register_task(name):
    def decorator(func):
        task_registry[name] = func
        return func
    return decorator

# Example task implementations
@register_task("lesion")
def segment_lesion() -> Dict[str, Any]:
    print("[INFO] Performing lesion segmentation...")
    return {"segmentation": "lesion_mask.png"}

@register_task("cancer")
def diagnose_cancer() -> Dict[str, Any]:
    print("[INFO] Performing cancer diagnosis...")
    return {"diagnosis": "malignant"}

# Example DSL script
script = """
prompt Diagnose(im_id, region, modality)
    task segment::'lesion'()
    task diagnose::'cancer'()

action DiagnoseProstateCancer(case_id)
    model = load_model('llama-7b')
    run model on Diagnose(case_id, 'breast', 'us')

for:case_id in [1,2,3,4,5]:DiagnoseProstateCancer(case_id)
"""

# Run the DSL interpreter
mp = MediPrompt()
mp.parse_and_execute(script)
