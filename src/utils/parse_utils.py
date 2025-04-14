def append_text_to_prompt(prompt_form, text):
    if len(prompt_form) == 0:
        return text
    else:
        return prompt_form + " " + text