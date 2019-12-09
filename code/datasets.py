from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import glob
from pathlib import Path
import json
import h5py
import re
import random

from code.config import cfg


class ClevrDataset(data.Dataset):
    def __init__(self, data_dir, split='train'):

        with open(os.path.join(data_dir, '{}.pkl'.format(split)), 'rb') as f:
            self.data = pickle.load(f)
        self.img = h5py.File(os.path.join(data_dir, '{}_features.h5'.format(split)), 'r')['features']

    def __getitem__(self, index):
        imgfile, question, answer, family = self.data[index]
        id = int(imgfile.rsplit('_', 1)[1][:-4])
        img = torch.from_numpy(self.img[id])

        return img, question, len(question), answer, family

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    images, lengths, answers, _ = [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = batch
    # sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer, family = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)

    return {'image': torch.stack(images), 'question': torch.from_numpy(questions),
            'answer': torch.LongTensor(answers), 'question_length': lengths}

class ClevrDialogDataset(ClevrDataset):
    def __init__(self, data_dir, split='train'):
        super(ClevrDialogDataset, self).__init__(data_dir, split)
    
    def __getitem__(self, index):
        imgfile, questions, answers, templates = self.data[index]
        id = int(imgfile.rsplit('_', 1)[1][:-4])
        img = torch.from_numpy(self.img[id])

        return img, questions, [len(q) for q in questions], answers, templates
    
    @staticmethod
    def collate_fn(batch):
        images = []
        batch_size = len(batch)

        max_len = -1
        for b in batch:
            max_len = max(max_len, max(b[2]))

        questions = np.zeros((10, batch_size, max_len), dtype=np.int64)
        question_lens = np.zeros((10, batch_size))
        answers = np.zeros((10, batch_size), dtype=np.int64)

        for i,b in enumerate(batch):
            img, qs, lens, ans, _ = b
            images.append(img)

            for j, (q, l, a) in enumerate(zip(qs, lens, ans)):
                questions[j, i, :l] = q
                question_lens[j, i] = l
                answers[j, i] = a
        
        return {'image': torch.stack(images), 'question': torch.from_numpy(questions),
                'answer': torch.LongTensor(answers.flatten()), 'question_length': torch.LongTensor(question_lens)}