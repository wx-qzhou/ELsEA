import re
import json
import opencc
import codecs
import multiprocessing
from multiprocessing import Pool
from os.path import join, dirname
from deep_translator import GoogleTranslator
from transformers import MarianMTModel, MarianTokenizer

# loads make str to dict
def load_json(rffile):
    with codecs.open(rffile, 'r', encoding='utf-8') as rf:
        return json.load(rf)

# dumps makes dict to str
def dump_json(obj, wffile, indent=4):
    with codecs.open(wffile, 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)

def translate_to_english(text):
    translator = GoogleTranslator(source='chinese (simplified)', target='english')
    return translator.translate(text)

def is_english_expression(text):
    english_chars = sum(c.isascii() or c.isalpha() for c in text)
    total_chars = len(text)
    return total_chars - english_chars == 0 if total_chars > 0 else False

def TraditionaltoSimplified(ent_id2uri):
    converter = opencc.OpenCC('t2s')  # 't2s' 表示从繁体到简体
    for idx in range(0, len(ent_id2uri)):
        ent_id2uri[idx][1] = converter.convert(ent_id2uri[idx][1])
    return ent_id2uri

# Remove the repeated words
def advanced_remove_repeats(text):
    pattern = re.compile(r'(\w+_\W+)\1+')
    cleaned_text = pattern.sub(r'\1', text)
    return cleaned_text

# translate other languages into English
def translate_item(item):
    idx, text = item
    if int(idx) % 10000 == 0:
        print("Now is {}.".format(idx))
    http = "/".join(text.split("/")[:-1])
    inputs = tokenizer(text.split("/")[-1], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_text = http + "/" + advanced_remove_repeats(tokenizer.decode(translated[0], skip_special_tokens=True).replace(" ", "_"))
    return idx, translated_text

# the main function of Helsinki_NLP
def Helsinki_NLP(text_list=[[0, "こんにちは"]], model_name='opus-mt-en-zh', eeflag=True):
    print(model_name)
    print(eeflag)
    
    text_list1 = []
    text_list2 = []
    if eeflag:
        for idtext in text_list:
            idx, text = idtext
            if is_english_expression(text):
                text_list1.append([idx, text])
            else:
                text_list2.append([idx, text])
    print("The number of no english expression is {}.".format(len(text_list2)))

    global tokenizer, model
    tokenizer = MarianTokenizer.from_pretrained(join(dirname(__file__), model_name))
    model = MarianMTModel.from_pretrained(join(dirname(__file__), model_name))

    cpu = min(multiprocessing.cpu_count(), 20)
    # cpu = multiprocessing.cpu_count()
    print('cpu has', cpu, 'cores')

    # 使用多线程进行翻译
    with Pool(processes=cpu) as pool:
        results = pool.map(translate_item, text_list2)

    print("The number of english expression is {}.".format(len(text_list1)))
    print("The number of no english expression is {}.".format(len(results)))
    assert len(text_list) == len(text_list1) + len(results)
    text_list = dict(list(results) + text_list1)

    del results, text_list2, text_list1
    return text_list