import os
import sys
import json
import pickle

import nltk
from tqdm import tqdm
from PIL import Image
from copy import deepcopy

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

def process_question(root, split, word_dic=None, answer_dic=None, concat_ctx=False):
    if word_dic is None:
        word_dic = {'pad':0}

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
            
            questions = []
            answers = []
            question_temps = []

            if concat_ctx:
                qtoks = ctoks
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
                if concat_ctx:
                    qtoks.extend([word_dic.setdefault(w, len(word_dic)) for w in qwords])
                else:
                    qtoks = [word_dic.setdefault(w, len(word_dic)) for w in qwords]
                
                    if questions == []:
                        qtoks = ctoks + qtoks
                
                atok = word_dic.setdefault(answer, len(word_dic))                
           
                answer_idx = answer_dic.setdefault(answer, len(answer_dic))
                
                questions.append(deepcopy(qtoks))
                answers.append(answer_idx)
                question_temps.append(question_temp_idx)
                
                if concat_ctx:
                    qtoks.append(atok)
            result.append((img_filename, questions, answers, question_temps))
    with open('{}/{}.pkl'.format(data_dir, split), 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic

if __name__ == '__main__':
    root = '/home/mshah1/workhorse3/clevr-dialog/data'
    data_dir = '../data/wHistory-keepTurns/'
    concat_ctx = True
    if concat_ctx:
        data_dir = '../data/wHistory-keepTurns-concatCtx/'
    word_dic, answer_dic = process_question(root, 'train', concat_ctx=concat_ctx)
    process_question(root, 'val', word_dic, answer_dic, concat_ctx=concat_ctx)    

    dict_dir = os.path.join(data_dir, 'dic.pkl')
    with open(dict_dir, 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)
        
    if os.path.exists(os.path.join(data_dir, 'dic.pkl')):        
        with open(os.path.join(data_dir, 'dic.pkl'), 'rb') as f:
            dicts = pickle.load(f)
        process_question(root, 'test', dicts['word_dic'], dicts['answer_dic'], concat_ctx=concat_ctx)