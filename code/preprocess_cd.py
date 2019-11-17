import os
import sys
import json
import pickle

import nltk
from tqdm import tqdm
from PIL import Image

question_templates= [
      "seek-attr-imm",
      "seek-attr-imm2",
      "seek-attr-early",
      "seek-attr-sim-early",
      "seek-attr-rel-imm",
      "seek-attr-rel-early",
      "count-all",
      "count-other",
      "count-all-group",
      "count-attribute",
      "count-attribute-group",
      "count-obj-rel-imm",
      "count-obj-rel-imm2",
      "count-obj-rel-early",
      "count-obj-exclude-imm",
      "count-obj-exclude-early",

      "extreme-loc",
      "obj-relation",
      "random-obj",
      "unique-obj",
      "count-att",
    ]

VAL_SIZE = 1000

def process_question(root, split, word_dic=None, answer_dic=None):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}
        
    split_ = 'train' if split in ['train', 'val'] else 'val'
    with open(os.path.join(root, 'clevr_%s_raw.json' % split_)) as f:
        data = json.load(f)

    if split == 'train':
        data = data[VAL_SIZE:]
    elif split == 'val':
        data = data[:VAL_SIZE]    

    result = []
    word_index = 1
    answer_index = 0

    for img_datum in tqdm(data):
        img_filename = img_datum['image_filename']
        for dialog_datum in img_datum['dialogs']:
            caption = dialog_datum['caption']
            cwords = nltk.word_tokenize(caption)
            ctoks = [word_dic.setdefault(w, len(word_dic)) for w in cwords]
            for turn in dialog_datum['dialog']:
                question = turn['question']
                answer = turn['answer']

                if turn['template'] in question_templates:
                    question_temp_idx = question_templates.index(turn['template'])
                else:
                    question_templates.append(turn['template'])
                    question_temp_idx = len(question_templates) - 1

                assert question_temp_idx >= 0

                qwords = nltk.word_tokenize(question)  
                
                qtoks = [word_dic.setdefault(w, len(word_dic)) for w in qwords]
           
                answer_idx = answer_dic.setdefault(answer, len(answer_dic))

                result.append((img_filename, qtoks, answer_idx, question_temp_idx))

    with open('../data/{}.pkl'.format(split), 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic

if __name__ == '__main__':
    root = sys.argv[1]

    # word_dic, answer_dic = process_question(root, 'train')
    # process_question(root, 'val', word_dic, answer_dic)    

    # with open('../data/dic.pkl', 'wb') as f:
    #     pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)
    
    if os.path.exists('../data/dic.pkl'):
        with open('../data/dic.pkl', 'rb') as f:
            dicts = pickle.load(f)
        process_question(root, 'test', dicts['word_dic'], dicts['answer_dic'])