import re
import os
import json

# load data
data_path = '../data/2023-04-12_oasst_ready.trees.jsonl'
with open(data_path, 'r') as f:
    data = [json.loads(line) for line in f]

def extract_code_block(text):
    code = re.findall(r'```[a-z]*\n([\s\S]*?)\n```', text, re.MULTILINE)
    return code if len(code) > 0 else []

def find_code_blocks(tree):
    code_blocks = []
    message = tree['prompt']
    code_blocks.extend(extract_code_block(message['text']))
    while len(message['replies']) > 0:
        message = message['replies'][0]
        code_blocks.extend(extract_code_block(message['text']))
    return code_blocks

# find all trees that contain code blocks
code_trees = []
for tree in data:
    if len(find_code_blocks(tree)) > 0:
        code_trees.append(tree['message_tree_id'])

# check if these trees have been translated
for tree in code_trees:
    if os.path.exists('../data/google_translate/' + tree + '.json'):
        # remove the translated tree file to force re-translation
        os.remove('../data/google_translate/' + tree + '.json')
