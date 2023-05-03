import os
import re
import json

# data preprocessing
CODE_TRANSLATION_CONSTANT = 'XXX123XXX'

def separate_text_and_code(text):
    code_blocks = extract_code_block(text)
    for code in code_blocks:
        text = text.replace(code, CODE_TRANSLATION_CONSTANT)
    return text, code_blocks

def extract_code_block(text):
    code = re.findall(r'```[a-z]*\n([\s\S]*?)\n```', text, re.MULTILINE)
    return code if len(code) > 0 else []

# translation
def translate_message(message, translate):
    text = message['text']
    translation = translate(text)
    if translation is None or len(translation) == 0:
        print('Translation failed for:', text)
        message['translation'] = text
        return message
    message['translation'] = translation
    for reply in message['replies']:
        reply = translate_message(reply, translate)
    return message


def translate_tree(tree, translate):
    tree['prompt'] = translate_message(tree['prompt'], translate)
    return tree


def translate_dataset(data, translate, translations_path='data/translated', language='en'):
    failed_ids = []
    for tree in data:
        if tree['prompt']['lang'] != language:
            continue
        # catch errors at this stage so as not to stop translation of other trees
        tree_path = os.path.join(translations_path, tree['message_tree_id'] + '.json')
        if os.path.exists(tree_path):
            continue
        try:
            translated = translate_tree(tree, translate)
            # save to file
            with open(tree_path, 'w', encoding='utf-8') as f:
                json.dump(translated, f, ensure_ascii=False)
        except Exception as e:
            print(e)
            print('Translation failed for tree: ' + tree['message_tree_id'])
            failed_ids.append(tree['message_tree_id'])
            continue
    return failed_ids


# storing data
def combine_translations(translations_path):
    files = os.listdir(translations_path)
    translations = []
    for file in files:
        with open(os.path.join(translations_path, file), 'r') as f:
            translations.append(json.load(f))
    return translations

def save_translations(translations_path, file_path):
    translations = combine_translations(translations_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        for translation in translations:
            json.dump(translation, f, ensure_ascii=False)
            f.write('\n')
