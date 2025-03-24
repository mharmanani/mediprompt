import re
import os

data_registry = {}
task_registry = {}
CONTEXT = {}

def append_text_to_prompt(prompt_form, text):
    if len(prompt_form) == 0:
        return text
    else:
        return prompt_form + " " + text

def fill_prompt_form(prompt_name, prompt_form, params):
    prompt_tokens = prompt_form.split(" ")
    i = 1 # 1-indexed to skip image id
    for token in prompt_tokens:
        if token.startswith("<") and token.endswith(">"):
            if CONTEXT[prompt_name][i] == token[1:-1]:
                curr_param = params[i]
                if curr_param.startswith("'") and curr_param.endswith("'"):
                    curr_param = curr_param[1:-1]
                prompt_form = prompt_form.replace(token, curr_param)
            i += 1
        else:
            continue
    
    if data_registry != {}:
        keys = list(data_registry.keys())
        for k in keys:
            data_registry.pop(k)
    
    if 3 in CONTEXT[prompt_name]: # check presence of data
        for (k,v) in CONTEXT[prompt_name][3].items():
            data_registry[k] = v
    
    return {params[0]: prompt_form}

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

                if len(params) > 0:
                    try:
                        assert 'im_id' in CONTEXT[prompt_name].values()
                        assert 'region' in CONTEXT[prompt_name].values()
                        assert 'modality' in CONTEXT[prompt_name].values()

                    except AssertionError:
                        #print(f"Prompt {prompt_name} must have parameters 'im_id', 'region', and 'modality'")
                        raise SyntaxError(f"Prompt {prompt_name} must have parameters 'im_id', 'region', and 'modality'")

            prompt_form = ''             
            while 'endprompt' not in line:
                line = lines[line_no+1]
                if line.strip().startswith("pathology"):
                    match = re.match(r"pathology: \s*'([^']*)'", line.strip())
                    if match:
                        params = CONTEXT[prompt_name].values()
                        try:
                            im_id = CONTEXT[prompt_name][0]
                            region = CONTEXT[prompt_name][1]
                            modality = CONTEXT[prompt_name][2]
                            pathology = match.groups()[0]                        
                            msg = f"Diagnose <{region}> {pathology} in this <{modality}> image."
                            prompt_form = append_text_to_prompt(prompt_form, msg)
                            prompts[prompt_name] = prompt_form
                        except KeyError:
                            raise SyntaxError(f"Must have parameter 'region' to specify pathology")
                
                if line.strip().startswith("target"):
                    match = re.match(r"target: \s*'([^']*)'", line.strip())
                    if match:
                        im_id = CONTEXT[prompt_name][0]
                        region = CONTEXT[prompt_name][1]
                        modality = CONTEXT[prompt_name][2]
                        target = match.groups()[0]                        
                        msg = f"Segment <{region}> {target} in this <{modality}> image."
                        prompt_form = append_text_to_prompt(prompt_form, msg)
                        prompts[prompt_name] = prompt_form

                if line.strip().startswith("data"):
                    match = re.match(r"data: \s*([^']*)", line.strip())
                    if match:
                        dataset = match.groups()[0]
                        idx = 0
                        data_map = {}
                        for item in os.listdir(f'./data/{dataset}'):
                            data_map[idx] = f'./data/{dataset}/{item}'
                            idx += 1
                        CONTEXT[prompt_name][3] = data_map
                        print(CONTEXT[prompt_name][3])

                if line.strip().startswith("text"):
                    match = re.match(r"text: \s*'([^']*)'", line.strip())
                    if match:
                        text = match.groups()[0]
                        prompt_form = append_text_to_prompt(prompt_form, text)
                        prompts[prompt_name] = prompt_form
                
                line_no += 1
        
        elif line.startswith("task"):
            match = re.match(r"task\s+(\w+)::'([^']*)'\(\)", line)
            if match:
                task_type, task_name = match.groups()
                if task_name in task_registry:
                    result = task_registry[task_name]()
                    CONTEXT[task_type] = result
                    commands.append(f"Execute task: {task_name}, result: {result}")

        elif line.startswith("data"):
            match = re.match(r"data\s+(\w+)::'([^']*)'\(\)", line)
            if match:
                print(data_type, data_name)
                data_type, data_name = match.groups()
                if data_name in task_registry:
                    result = task_registry[data_name]()
                    CONTEXT[data_type] = result
                    commands.append(f"Execute data: {data_name}, result: {result}")

        elif line.startswith("show"): # equivalent to printing the prompt
            tokens = line.strip().split(" ")
            if len(tokens) != 2:
                raise SyntaxError("Show command must have exactly 2 tokens")
            else:
                prompt_name = tokens[1]
                if prompt_name in prompts:
                    commands.append(f"print:{prompts[prompt_name]}")
                else:
                    raise ValueError(f"Prompt {prompt_name} not found in prompts")

        elif line.startswith("run"):
            # need group forn 'run' token, <model> token, then a prompt (e.g. Diagnose(...))
            match = re.match(r"run (\w+) on (\w+)\(([^)]*)\)", line)
            if match:
                model, prompt_name, params = match.groups()
                params = [p.strip().rstrip() for p in params.split(",")]
                prompt = fill_prompt_form(prompt_name, prompts[prompt_name], params)
                prompt_img = data_registry[int(list(prompt.keys())[0])]
                prompt_text = list(prompt.values())[0]
                commands.append(f"run:{model}:{prompt_img}:{prompt_text}")
            else:
                raise SyntaxError("Run command must have the format 'run <model> on <prompt>'")

    return prompts, commands, data_registry

#prompts, commands = parse_script(script)
#print(prompts)
#print(commands)