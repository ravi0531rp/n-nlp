import json


def get_openai_key(path):
    with open(path, "r") as f:
        return json.load(f)["key"]
