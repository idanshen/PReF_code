import numpy as np
import itertools
import json
import os
import random

from pairs.completions import get_completion
import pairs.prompts as prompts

attribute_labels = {
    "length": "How long and detailed the responses are.",
    "formality": "How formal the responses are.",
    "humour": "How humorous the responses are.",
    "elicitation": "Does the model ask a lot of follow up questions?",
    "enthusiasm": "How enthusiastic the responses are.",
    "politeness": "How polite the model is.",
    "confidence": "How confident the model appears in its responses.",
}

attributes = {
    "length": {"a": "verbose", "b": "concise"},
    "formality": {"a": "formal", "b": "informal"},
    "humour": {"a": "humorous", "b": "serious"},
    "elicitation": {"a": "engaging", "b": "unengaging"},
    "politeness": {"a": "polite", "b": "rude"},
    "enthusiasm": {"a": "enthusiastic", "b": "demure"},
    "confidence": {"a": "confident", "b": "uncertain"},
}


def get_users_enums(with_bad_users=False):
    predicates = [(attr, pred) for attr in attributes for pred in ("a", "b")]
    predicate_pairs = list(itertools.combinations(predicates, 2))
    users = [
        (attr1, attr2, dir1, dir2) for (attr1, dir1), (attr2, dir2) in predicate_pairs
    ]
    if not with_bad_users:
        bad_users = [i for i, user in enumerate(users) if user[0] == user[1]]
        users = [user for i, user in enumerate(users) if i not in bad_users]
    return users


def sample_user(model_dir, users):
    test_users_path = os.path.join(model_dir, "test_users.json")
    with open(test_users_path, "r") as file:
        test_users_indices = json.load(file)
    test_users = [users[i] for i in test_users_indices]
    user = random.sample(test_users, 1)
    return user


def get_preference(instruction, resp1, resp2, attr1, attr2, dir1, dir2):
    if any([inp is None for inp in (attr1, attr2, dir1, dir2)]):
        sys_prompt = f"You are a helpful AI judge."
    else:
        sys_prompt = f"You are a helpful AI judge. You prefer {attributes[attr1][dir1]} and {attributes[attr2][dir2]} responses."
    prompt1 = prompts.basic_choice_no_reason.format(
        instruction=instruction, output_1=resp1, output_2=resp2
    )
    prompt2 = prompts.basic_choice_no_reason.format(
        instruction=instruction, output_1=resp2, output_2=resp1
    )
    pref1 = get_completion(prompt1, system_prompt=sys_prompt, temp=0.0)
    pref2 = get_completion(prompt2, system_prompt=sys_prompt, temp=0.0)
    return pref1, pref2


def get_preference_prism(user_description, prompt, response_1, response_2):
    prompt1 = prompts.PRISM_no_confidence.format(
        prompt=prompt,
        user_description=user_description,
        response_1=response_1,
        response_2=response_2,
    )
    prompt2 = prompts.PRISM_no_confidence.format(
        prompt=prompt,
        user_description=user_description,
        response_1=response_2,
        response_2=response_1,
    )
    pref1 = get_completion(prompt1, system_prompt=None, model="gpt-4o-mini", temp=0.0)
    pref2 = get_completion(prompt2, system_prompt=None, model="gpt-4o-mini", temp=0.0)
    return pref1, pref2


def get_pref(input):
    instruction, resp1, resp2, user = input
    pref1, pref2 = get_preference(instruction, resp1, resp2, *user)
    return pref1, pref2


def get_pref_prism(input):
    instruction, resp1, resp2, user = input
    pref1, pref2 = get_preference_prism(user, instruction, resp1, resp2)
    return pref1, pref2


def to_preference(text):
    if text == ("Output (a)", "Output (b)"):
        return 1
    elif text == ("Output (b)", "Output (a)"):
        return 0
    elif text == ("Output (a)", "Output (a)"):
        return 0.5
    elif text == ("Output (b)", "Output (b)"):
        return 0.5
    else:
        raise ValueError


def to_preference_prism(pref1, pref2):
    if "A" in pref1 and "B" in pref2:
        return 1
    elif "B" in pref1 and "A" in pref2:
        return 0
    else:
        return 0.5


def parse_attribute_entries(filename):
    entries = {}
    current_keyword = None
    current_items = []

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line.startswith("**") and line.endswith("**"):
                # Save the previous keyword and its items
                if current_keyword is not None:
                    entries[current_keyword] = current_items

                # Start a new keyword entry
                current_keyword = line.strip("**").strip(":")
                current_items = []

            elif line and line[0].isdigit() and "." in line:
                # Extract the item text after the numbering
                item_text = line.split(".", 1)[1].strip()
                current_items.append(item_text)

        # Save the last keyword entry
        if current_keyword is not None:
            entries[current_keyword] = current_items

    return entries
