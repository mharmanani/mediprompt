from src.mediprompt import MediPrompt
from src.utils import sysutils as S

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

script = open(args.script).read()

prompts, commands, data = parse_script(script)

for command in commands:
    cmd, *args = command.split(":")
    S.exec_(cmd, args, data)