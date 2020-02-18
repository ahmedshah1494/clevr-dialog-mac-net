import os
import sys
import json
import pickle

import nltk
import tqdm
from PIL import Image

VAL_SIZE = 50000

def process_question(root, split, word_dic=None, answer_dic=None):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    # split_ = 'train' if split in ['train', 'val'] else 'val'
    with open(os.path.join(root, 'questions', 'CLEVR_{}_questions.json'.format(split))) as f:
        data = json.load(f)

    # if split == 'train':
    #     data['questions'] = data['questions'][VAL_SIZE:]
    # elif split == 'val':
    #     data['questions'] = data['questions'][:VAL_SIZE]

    result = []
    word_index = 1
    answer_index = 0

    for question in tqdm.tqdm(data['questions']):
        words = nltk.word_tokenize(question['question'])
        question_token = []

        for word in words:
            try:
                question_token.append(word_dic[word])

            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        answer_word = question['answer']

        try:
            answer = answer_dic[answer_word]

        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1

        result.append((question['image_filename'], question_token, answer, question['question_family_index']))

    with open('{}/{}.pkl'.format(data_dir, split), 'wb') as f:
        pickle.dump(result, f)

    return word_dic, answer_dic

if __name__ == '__main__':
    root = sys.argv[1]
    data_dir = '/home/mshah1/workhorse3/clevr-dialog/macnet-data/clevr/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    word_dic, answer_dic = process_question(root, 'train')
    process_question(root, 'val', word_dic, answer_dic)

    with open(os.path.join(data_dir, 'dic.pkl'), 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)