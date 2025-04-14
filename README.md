# MediPrompt: A Domain Specific Language for Prompting Medical Foundation Models
**MediPrompt** is a domain-specific language (DSL) for building and executing structured prompts for vision-language models (VLMs) in clinical imaging tasks. It allows users—especially those without programming or AI expertise—to define reusable, interpretable, and data-aware prompts for medical image analysis.

## Features

- ✅ Simple syntax for defining prompts  
- ✅ Support for multiple clinical tasks: diagnosis, segmentation, description  
- ✅ Built-in support for data loading and formatting  
- ✅ Model registry and execution engine  
- ✅ Templated prompt generation with semantic structure  
- ✅ Works with any VLM (e.g., Qwen, LLaMA, GPT-4V)

## Getting Started

### Example Prompts

```plaintext
prompt HelloWorld()
    text: 'Hello, world!'
endprompt

show HelloWorld()
```
This program should output a simple `'Hello, world!'` message to the screen. To try it, running `./mediprompt.sh prompts/helloworld.mediprompt`.

```plaintext
export report.log
prompt AnalyzeCXR(id, rgn, mod)
    data: CXR
    context
    describe
endprompt  

show AnalyzeCXR(0, 'chest', 'xray')
```

This program generates a descriptive prompt for a chest X-ray image. When run with a VLM, it produces a structured diagnostic report. Try running `./mediprompt.sh prompts/cxr.mediprompt`.

## Syntax Overview

MediPrompt supports the following core statements:

- `prompt <name>(args) { ... }`: Defines a named prompt template  
- `data: <path>`: Binds a dataset to the prompt  
- `context`: Activates contextual information mode  
- `reasoning`: Enables chain-of-thought prompting  
- `text: '<string>'`: Appends arbitrary user-defined text  
- `pathology: '<term>'`: Instantiates a diagnostic prompt template  
- `for all <var>` or `for <var>:<start>..<end>`: Loop over images  
- `run <model> on <prompt_call>`: Executes prompt with a specified model  
- `show <prompt_call>`: Outputs the generated prompt text  

## Example Use Cases

- 🧬 Cancer detection in histopathology  
- 🩺 Breast cancer screening with ultrasound  
- 🫁 Chest X-ray report generation  

See the `prompts/` folder for complete prompts and `logs/` for outputs.

## Architecture

- **Parser**: Converts MediPrompt scripts into an internal AST  
- **Executer**: Evaluates prompt statements and handles iteration  
- **Model Registry**: Manages connections to available VLMs  
- **Prompt Generator**: Constructs natural language prompts  
- **Output Logger**: Records sessions, predictions, and timestamps  

## Citation

If you use MediPrompt in your research, please cite:

```
@inprogress{mediprompt2025,
  title = {MediPrompt: A Prompt Programming Language for Clinical Vision-Language Tasks},
  author = {Author et al.},
  year = {2025},
  journal = {In submission}
}
```

## License

MIT License

---

Let me know if you'd like a shorter version or one targeted at a different audience (e.g. for clinicians, developers, or researchers).
