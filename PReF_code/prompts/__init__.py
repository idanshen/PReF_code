import os
import pdb

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, 'basic_choice.txt')) as f:
    basic_choice_prompt = f.read()

with open(os.path.join(dir_path, 'basic_choice_no_reason.txt')) as f:
    basic_choice_no_reason = f.read()

with open(os.path.join(dir_path, 'diana.txt')) as f:
    diana_persona = f.read()

with open(os.path.join(dir_path, 'joe.txt')) as f:
    joe_persona = f.read()

with open(os.path.join(dir_path, 'persona_gen.txt')) as f:
    persona_gen = f.read()

with open(os.path.join(dir_path, 'PRISM_get_preference.txt')) as f:
    PRISM_get_preference = f.read()

with open(os.path.join(dir_path, 'PRISM_no_confidence.txt')) as f:
    PRISM_no_confidence = f.read()
