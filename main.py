from src.mediprompt import MediPrompt

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
