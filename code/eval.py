import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import pickle
import sys
import os
from tqdm import tqdm
from multiprocessing import cpu_count

from code.datasets import ClevrDataset, ClevrDialogDataset, collate_fn
from code.config import cfg, cfg_from_file
from code.utils import load_vocab
from code.mac import MACNetwork

import argparse

def load_model(cpfile, return_attn_maps=False):
    cpdata = torch.load(cpfile)
    cpdir = os.path.join(os.path.dirname(cpfile), '../')
    [cfgfile] = [fn for fn in os.listdir(cpdir) if 'yml' == fn.split('.')[-1]]
    cfgfile = os.path.join(cpdir, cfgfile)
        
    cfg_from_file(cfgfile)
    
    if return_attn_maps:        
        cfg.EVAL.RETURN_ATTENTION_MAPS = return_attn_maps
        print(cfg.EVAL.RETURN_ATTENTION_MAPS)

    vocab = load_vocab(cfg)
    
    model = MACNetwork(cfg, cfg.TRAIN.MAX_STEPS, vocab)    
    model.load_state_dict(cpdata['model'])
   
    if cfg.CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    if cpdata['optim'] is not None:
        optimizer.load_state_dict(cpdata['optim'])        
    return model, vocab, optimizer

def evaluate(args):
    model, vocab, _ = load_model(args.checkpoint_file, args.save_attention_maps)
    model.eval()
    if cfg.CUDA:
        model = model.cuda()
    
    if cfg.TRAIN.CLEVR_DIALOG:
        dataset_test = ClevrDialogDataset(data_dir=args.data_dir, split="test")
        batch_size = 128
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, drop_last=False,
                                            shuffle=False, num_workers=cfg.WORKERS, collate_fn=ClevrDialogDataset.collate_fn)
    else:
        dataset_test = ClevrDataset(data_dir=args.data_dir, split='test')
        batch_size = 512
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, drop_last=False,
                                    shuffle=False, num_workers=cpu_count(), collate_fn=collate_fn)

    total_correct = 0
    total = 0
    predictions = []
    gt_answers = []
    program_attn_maps = []
    image_attn_maps = []
    control_history = []

    t = tqdm(dataloader_test)
    for data in t:
        image, question, question_len, answer = data['image'], data['question'], data['question_length'], data['answer']
        answer = answer.long()
        question = Variable(question)
        answer = Variable(answer)

        if cfg.CUDA:
            image = image.cuda()
            question = question.cuda()
            answer = answer.cuda().squeeze()

        with torch.no_grad():
            if cfg.EVAL.RETURN_ATTENTION_MAPS:
                scores, ch, ch_attn_map, know_attn_map = model(image, question, question_len)                

                if ch.shape[0] > batch_size:                
                    ch = ch.detach().cpu().numpy().reshape(10, -1, 8, 512).transpose((1,0,2,3))
                    control_history.append(ch)

                    ch_attn_map = ch_attn_map.detach().cpu().numpy()
                    program_attn_maps.append(ch_attn_map)

                    know_attn_map = know_attn_map.detach().cpu().numpy()
                    image_attn_maps.append(know_attn_map)
                else:
                    control_history.append(ch)

            else:
                scores = model(image, question, question_len)        

        preds = scores.detach().argmax(1)        
        correct = preds == answer
        correct = correct.sum().cpu().numpy()
        
        total_correct += correct
        total += answer.shape[0]
        
        if cfg.TRAIN.CLEVR_DIALOG:
            preds = preds.view(10, -1).transpose(0,1).contiguous().view(-1)
            answer = answer.view(10, -1).transpose(0,1).contiguous().view(-1)

        predictions.append(preds.cpu().numpy())        
        gt_answers.append(answer.detach().cpu().numpy())
        t.set_postfix(acc=(total_correct/total))        

    print(total_correct, total)
    accuracy = total_correct / total

    if args.save_attention_maps:
        print(len(control_history), len(program_attn_maps), len(image_attn_maps))
        control_history = np.concatenate(control_history, axis=0)        
        outfile = args.checkpoint_file.replace('pth', 'control_history.npy')
        np.save(outfile, control_history)

        if len(program_attn_maps) > 0:
            program_attn_maps = np.concatenate(program_attn_maps, axis=0)        
            outfile = args.checkpoint_file.replace('pth', 'program_attention.npy')
            np.save(outfile, program_attn_maps)

        if len(image_attn_maps) > 0:
            image_attn_maps = np.concatenate(image_attn_maps, axis=0)
            outfile = args.checkpoint_file.replace('pth', 'image_attention.npy')
            np.save(outfile, image_attn_maps)

    predictions = np.concatenate(predictions, axis=0)
    gt_answers = np.concatenate(gt_answers, axis=0)
    
    outfile = args.checkpoint_file.replace('pth', 'preds')
    np.savetxt(outfile, predictions, fmt='%d')

    predictions = [vocab['answer_idx_to_token'][a] for a in predictions]
    outfile = args.checkpoint_file.replace('pth', 'preds.txt')
    np.savetxt(outfile, predictions, fmt='%s')

    outfile = args.checkpoint_file.replace('pth', 'answers')
    np.savetxt(outfile, gt_answers, fmt='%d')

    gt_answers = [vocab['answer_idx_to_token'][a] for a in gt_answers]
    outfile = args.checkpoint_file.replace('pth', 'answers.txt')
    np.savetxt(outfile, gt_answers, fmt='%s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_file', type=str)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--save_attention_maps', action='store_true')
    args = parser.parse_args()

    evaluate(args)