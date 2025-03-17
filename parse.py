"""
INPUT STRING:
prompt Diagnose(im_id, region, modality)
    task segment::'lesion'()
    task diagnose::'cancer'()

run llama with Diagnose(2147, 'prostate', 'mri')
run llama with Diagnose(115, 'breast', 'ultrasound')

Actions:
    * Load image id 2147 into memory (using e.g. numpy, PIL, etc.)
    * Declare an empty list prompts=[]
    * Declare an empty list commands=[]
    * Declare a dictionary CONTEXT={}

    * Parse the 'prompt' line and store the prompt name and parameters in the CONTEXT dictionary
    * Parse the 'task' lines and execute the corresponding task function, storing the result in the CONTEXT dictionary
    * Parse the 'action' lines and execute the corresponding action function

Output:
prompts = {2147:"Diagnose prostate cancer in this MRI image of the prostate", 
            115:"Diagnose breast cancer in this ultrasound image"}

for id in (2147, 86):
    img = load_image(id)
    # run llama on img with prompt in prompts
    y = run_llama(img, prompt[img])
    print(y)
"""

script = """
prompt Diagnose(im_id, region, modality)
    pathology: 'cancer'
endprompt

prompt Segment(im_id, region, modality)
endprompt
    
for im_id=0..200
    run llama on Diagnose(2147, 'prostate', 'mri')
"""

import re

task_registry = {}
CONTEXT = {}

def parse_prompt(prompt_name, params):
    # look for prompt in the CONTEXT dictionary
    assert prompt_name in CONTEXT.keys()
    params_def_list = CONTEXT[prompt_name]
    prompt_instance = {}
    for p_id in params_def_list:
        var_name = params_def_list[p_id]
        var_value = params[p_id]
        if var_value.startswith("'") and var_value.endswith("'"):
            var_value = var_value[1:-1]
        prompt_instance[var_name] = var_value
    
    return f"{prompt_name} {prompt_instance['region']} cancer in this {prompt_instance['modality']} image"

def parse_script(script: str):
    prompts = {}
    commands = []
    lines = script.strip().split("\n")
    for line_no in range(len(lines)):
        line = lines[line_no].strip()
        
        if line.startswith("prompt"):
            match = re.match(r"prompt\s+(\w+)\(([^)]*)\)", line)
            if match:
                prompt_name, params = match.groups()
                CONTEXT[prompt_name] = {i:p.strip() for i, p in enumerate(params.split(","))}
            
            while 'endprompt' not in line:
                line = lines[line_no+1]
                print(line, 'endprompt' in line)
                if line.strip().startswith("\tpathology"):
                    match = re.match(r"\tpathology:\s*'([^']*)'", line)
                    if match:
                        pathology = match.groups()[0]
                        prompts[CONTEXT[prompt_name][0]] = f"Diagnose {pathology} cancer in this image"

                line_no += 1

            print(CONTEXT)
        
        elif line.startswith("task"):
            match = re.match(r"task\s+(\w+)::'([^']*)'\(\)", line)
            if match:
                task_type, task_name = match.groups()
                if task_name in task_registry:
                    result = task_registry[task_name]()
                    CONTEXT[task_type] = result
                    commands.append(f"Execute task: {task_name}, result: {result}")

        elif line.startswith("run"):
            # need group forn 'run' token, <model> token, then a prompt (e.g. Diagnose(...))
            match = re.match(r"run (\w+) on (\w+)\(([^)]*)\)", line)
            if match:
                model, prompt_name, params = match.groups()
                params = [p.strip().rstrip() for p in params.split(",")]
                prompt = parse_prompt(prompt_name, params)
                commands.append(f"Run model {model} with prompt {prompt}")

    return prompts, commands

prompts, commands = parse_script(script)
print(prompts)
print(commands)