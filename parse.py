import re
import os
import time

data_registry = {}
task_registry = {}
CONTEXT = {}

def append_text_to_prompt(prompt_form, text):
    if len(prompt_form) == 0:
        return text
    else:
        return prompt_form + " " + text

def get_case_ids(case_id_val):
    permitted_symbols = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
    try: 
        case_id_val = int(case_id_val)
        return case_id_val
    except ValueError: # string or list, handle
        for c in case_id_val:
            if c not in permitted_symbols:
                raise SyntaxError(f"Invalid name: {case_id_val}")

        if case_id_val in CONTEXT.keys():
            case_id_val = CONTEXT[case_id_val]
            # found it, but it could be a list or an int
            # if int, we return it
            if case_id_val == "!USEALL!":
                return list(range(len(data_registry)))
            if type(case_id_val) == int:
                return case_id_val
            # if list, we return the list, now we have to loop
            elif type(case_id_val) == list:
                return [int(x) for x in case_id_val]
            else:
                raise SyntaxError(f"Invalid case id value: {case_id_val}")
        
        elif len(case_id_val) == 0:
            return

        else:
            raise ValueError(f"Undefined variable: {case_id_val}")

    

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

def parse_script(script, 
                 CONTEXT=CONTEXT,
                 data_registry=data_registry,
                 task_registry=task_registry):
    prompts = {}
    commands = []
    lines = script.strip().split("\n")
    export_flag = False
    export_filename = ""
    for line_no in range(len(lines)):
        line = lines[line_no].strip()

        # Comments - skip
        if line.startswith(";") or line == "":
            continue
        
        # export command, create a log file
        if line.startswith("export"):
            match = re.match(r"export\s+(\w+)", line)
            if match:
                filename = match.groups()[0]
                export_flag = True
                export_filename = filename
                # create a log file
                try: 
                    os.stat(f'./logs')
                except FileNotFoundError:
                    os.makedirs(f'./logs', exist_ok=True)
                
                with open(f'./logs/{filename}.log', 'w') as f:
                    f.write(f"[{time.ctime()}] SESS START\n")
        
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
                            msg = f"Is there any {pathology} in this image?"
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

                if (line.strip().startswith("reasoning") and \
                    line.strip().endswith("reasoning")):
                    params = CONTEXT[prompt_name].values()
                    msg = "Please explain your reasoning."
                    prompt_form = append_text_to_prompt(prompt_form, msg)
                    prompts[prompt_name] = prompt_form

                if (line.strip().startswith("describe") and \
                    line.strip().endswith("describe")):
                    params = CONTEXT[prompt_name].values()
                    msg = "Describe the image. What do you conclude?"
                    prompt_form = append_text_to_prompt(prompt_form, msg)
                    prompts[prompt_name] = prompt_form


                if (line.strip().startswith("context") and \
                    line.strip().endswith("context")):
                    params = CONTEXT[prompt_name].values()
                    try:
                        im_id = CONTEXT[prompt_name][0]
                        region = CONTEXT[prompt_name][1]
                        modality = CONTEXT[prompt_name][2]
                        msg = f"This is a <{region}> <{modality}> image."
                        prompt_form = append_text_to_prompt(prompt_form, msg)
                        prompts[prompt_name] = prompt_form
                    except KeyError:
                        raise SyntaxError(f"Must have parameters 'region' and 'modality' to specify context")

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
                data_type, data_name = match.groups()
                if data_name in task_registry:
                    result = task_registry[data_name]()
                    CONTEXT[data_type] = result
                    commands.append(f"Execute data: {data_name}, result: {result}")

        elif line.startswith("show"): # equivalent to printing the prompt
            # must have the form 'show <prompt_name>' or 'show <prompt_name>(p0, p1, p2)'
            match = re.match(r"show\s+(\w+)(\(([^)]*)\))?", line)
            if match:
                prompt_name = match.groups()[0]
                params = match.groups()[2]
                if params:
                    params = [p.strip().rstrip() for p in params.split(",")]
                    if len(params) != 3:
                        raise SyntaxError("mismatch in the number of parameters")
                    prompt = fill_prompt_form(prompt_name, prompts[prompt_name], params)
                    prompt_text = list(prompt.values())[0]
                    if export_flag:
                        with open(f'./logs/{export_filename}.log', 'a') as f:
                            f.write(f"[{time.ctime()}] USER: {prompt_text}\n")
                    else:
                        commands.append(f"print:{prompt_text}")
                else:
                    prompt_text = prompts[prompt_name]
                    if export_flag:
                        with open(f'./logs/{export_filename}.log', 'a') as f:
                            f.write(f"[{time.ctime()}] USER: {prompt_text}\n")
                    else:
                        commands.append(f"print:{prompt_text}")
            else:
                raise SyntaxError("show command must have the format 'show <prompt_name>' or 'show <prompt_name>(p0, p1, p2)'")
                
        elif line.startswith("specify"): # save the prompt to a file
            # 'specify <dataset>'
            match = re.match(r"specify\s+(\w+)", line)
            if match:
                dataset = match.groups()[0]
                idx = 0
                data_map = {}
                for item in os.listdir(f'./data/{dataset}'):
                    data_map[idx] = f'./data/{dataset}/{item}'
                    idx += 1
                data_registry = data_map

        elif line.startswith("save"): # save the prompt to a file
            # 'save <prompt_name> <filename>'
            match = re.match(r"save\s+(\w+)\s+(\w+)", line)
            pass
                
        elif line.startswith("for"): # for loop
            match1 = re.match(r"for (\w+):(\w+)..(\w+)", line)
            match2 = re.match(r"for all (\w+)", line)
            if match1:
                loop_var, min_val, max_val = match1.groups()
                min_val = int(min_val)
                max_val = int(max_val)
                CONTEXT[loop_var] = list(range(min_val, max_val+1))
            elif match2:
                loop_var = match2.groups()[0]
                CONTEXT[loop_var] = "!USEALL!"
            else:
                raise SyntaxError(f"{line}'")

        elif line.startswith("run"):
            # need group forn 'run' token, <model> token, then a prompt (e.g. Detect(...))
            match = re.match(r"run (\w+) on (\w+)\(([^)]*)\)", line)
            if match:
                model, prompt_name, params = match.groups()
                params = [p.strip().rstrip() for p in params.split(",")]
                prompt = fill_prompt_form(prompt_name, prompts[prompt_name], params)
                case_ids = list(prompt.keys())
                case_ids = get_case_ids(case_ids[0])
                prompt_text = list(prompt.values())[0]
                if case_ids == '':
                    commands.append(f"run:{model}:{''}:{prompt_text}:{export_flag if export_flag else ''}:{export_filename}")
                if type(case_ids) == int:
                    prompt_img = data_registry[case_ids]
                    commands.append(f"run:{model}:{prompt_img}:{prompt_text}:{export_flag if export_flag else ''}:{export_filename}")
                elif type(case_ids) == list:
                    try: 
                        assert len(case_ids) <= len(data_registry)
                    except AssertionError:
                        if len(case_ids) > len(data_registry):
                            raise SyntaxError(f"Invalid number of cases")
                
                    for case_id in case_ids:
                        prompt_img = data_registry[case_id]
                        commands.append(f"run:{model}:{prompt_img}:{prompt_text}:{export_flag if export_flag else ''}:{export_filename}")
            else:
                raise SyntaxError("Run command must have the format 'run <model> on <prompt>'")

    return prompts, commands, data_registry

#prompts, commands = parse_script(script)
#print(prompts)
#print(commands)