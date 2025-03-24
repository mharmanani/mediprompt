from src.mediprompt import MediPrompt
from src.utils import sysutils as S

# load hello world script
script = open("example1.mediprompt").read()

# initialize the prompts list
prompts = []

# initalize commands list
commands = []

# Run the DSL interpreter
#mp = MediPrompt()
#mp.parse_and_execute(script)

from parse import parse_script

import argparse

# parse the script
argparser = argparse.ArgumentParser()

# accept 1 argument: the script file
argparser.add_argument("script", help="The script file to parse")
args = argparser.parse_args()

prompts, commands = parse_script(script)

for command in commands:
    print(f"Executing command: {command}")
    cmd, args = command.split(":")
    S.exec_(cmd, args)