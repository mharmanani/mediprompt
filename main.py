from src.mediprompt import MediPrompt

# load hello world script
script = open("helloworld.mediprompt").read()

# initialize the prompts list
prompts = []

# initalize commands list
commands = []

# Run the DSL interpreter
mp = MediPrompt()
mp.parse_and_execute(script)
