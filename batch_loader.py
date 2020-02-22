#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# import lib
import csv
import json
import os
from collections import OrderedDict
from random import shuffle

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm

from bert import BertTokenizer
from config.config import args

WHITESPACE_PLACEHOLDER = ' □ '


class BatchLoader(object):
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.nega_num = args.nega_num
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir,
                                                       do_lower_case=True)
        self.tag_to_ix = args.tag_to_ix

    def load_data(self, file_path):
        data = []
        num = 0
        with open(os.path.join(self.data_dir, file_path),
                  'r',
                  encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                data.append(sample)
        return data

    def build_data(self, data, is_train):
        '''
        build data list, split to train and dev
        tokenization
        '''
        ner_data = []
        re_data = []
        if is_train:
            for sample in data:
                # ner
                question = sample['question']
                s, p, o = sample['triple']
                ner_sample = self._get_ner_sample(question, s)
                ner_data.append(ner_sample)
                # re
                re_sample = self._get_re_sample(question, p, 1)
                re_data.append(re_sample)
                nps = sample.get('negative_predicates', [])
                shuffle(nps)
                for np in nps[:self.nega_num]:
                    re_sample = self._get_re_sample(question, np, 0)
                    re_data.append(re_sample)
        else:
            for sample in data:
                question = sample['question']
                s, p, o = sample['triple']
                ner_sample = self._get_ner_sample(question, s)
                ner_data.append(ner_sample)
        return ner_data, re_data

    def build_ner_data(self, question):
        '''
        build ner data for inference
        question: 
        '''
        ner_data = [self._get_ner_sample(question, None)]
        return ner_data

    def build_re_data(self, question, predicates):
        '''
        build re data for inference
        question: 
        predicates: all predicates relative to the subject
        '''
        re_data = []
        for p in predicates:
            re_sample = self._get_re_sample(question, p, 0)
            re_data.append(re_sample)
        return re_data

    def batch_loader(self,
                     ner_data=None,
                     re_data=None,
                     ner_max_len=64,
                     re_max_len=64,
                     batch_size=32,
                     is_train=True):
        if is_train:  # input all three datas
            ner_dataset = self._build_dataset(ner_data, None, ner_max_len)
            re_dataset = self._build_dataset(None, re_data, re_max_len)
            datasets = [ner_dataset, re_dataset]
            dataloaders = []
            for dataset in datasets:
                dataloaders.append(
                    DataLoader(dataset,
                               batch_size,
                               sampler=RandomSampler(dataset),
                               drop_last=True))
            return dataloaders
        else:  # only input ner_data
            if ner_data:
                ner_dataset = self._build_dataset(ner_data, None, ner_max_len)
                return DataLoader(dataset=ner_dataset,
                                  batch_size=batch_size,
                                  sampler=SequentialSampler(ner_dataset),
                                  drop_last=False)
            elif re_data:
                re_dataset = self._build_dataset(None, re_data, re_max_len)
                return DataLoader(dataset=re_dataset,
                                  batch_size=batch_size,
                                  sampler=SequentialSampler(re_dataset),
                                  drop_last=False)
            else:
                raise Exception('At least ner or re should not be None')

    def _build_dataset(self, ner_data=None, re_data=None, max_len=64):
        '''
        only input one data
        '''
        data, dataset = None, None
        if ner_data:
            data = ner_data
        elif re_data:
            data = re_data
        else:
            raise Exception('as least an input, ner or re')
        token_ids = torch.tensor(self._padding(
            [item['token_ids'] for item in data], max_len),
                                 dtype=torch.long)
        token_types = torch.tensor(self._padding(
            [item['token_types'] for item in data], max_len),
                                   dtype=torch.long)
        if ner_data:
            tag_list = torch.tensor(self._padding(
                [item['tags'] for item in data], max_len, is_label=True),
                                    dtype=torch.long)
            input_mask = torch.tensor(self._padding(
                [item['input_mask'] for item in data], max_len),
                                      dtype=torch.long)
            output_mask = torch.tensor(self._padding(
                [item['output_mask'] for item in data], max_len),
                                       dtype=torch.long)
            dataset = TensorDataset(token_ids, token_types, tag_list,
                                    input_mask, output_mask)
        elif re_data:
            label = torch.tensor([item['label'] for item in data],
                                 dtype=torch.float)
            dataset = TensorDataset(token_ids, token_types, label)
        return dataset

    def _get_token_ids(self, text, add_cls=False, add_sep=False, max_len=64):
        new_text = text.replace(' ', WHITESPACE_PLACEHOLDER).replace(
            '　', WHITESPACE_PLACEHOLDER)
        tokens = self.tokenizer.tokenize(new_text, inference=True)
        tokens = tokens[:max_len] if len(tokens) > max_len else tokens
        # BERT input need to add [CLS] and [SEP]
        tokens = ['[CLS]'] + (tokens[:-1] if len(tokens) +
                              1 > max_len else tokens) if add_cls else tokens
        tokens = (tokens[:-1] if len(tokens) +
                  1 > max_len else tokens) + ['[SEP]'] if add_cls else tokens
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens, token_ids

    def _get_ner_sample(self, question, subject):
        tokens, token_ids = self._get_token_ids(question, True, True)
        token_types = [0] * len(tokens)
        input_mask = [1] * len(tokens)
        ## Notes: label_id中不包括[CLS]和[SEP]
        tag_list = [self.tag_to_ix["O"] for i in range(len(tokens))]
        output_mask = [0] + input_mask[1:-1] + [0]
        if subject:
            span_tokens, _ = self._get_token_ids(subject)
            tag_list = self._get_tag_list(tokens, span_tokens, tag_list)
        ## Notes: label_id中不包括[CLS]和[SEP]
        tag_list = tag_list[1:-1]
        ner_sample = {
            'tokens': tokens,
            'token_ids': token_ids,
            'token_types': token_types,
            'tags': tag_list,
            'subject': subject,
            'input_mask': input_mask,
            'output_mask': output_mask
        }
        return ner_sample

    def _get_re_sample(self, question, predicate, label):
        p_tokens, p_token_ids = self._get_token_ids(predicate, True, True, 64)
        tokens, token_ids = self._get_token_ids(question, False, True)
        tokens, token_ids, token_types = p_tokens + tokens, p_token_ids + token_ids, [
            1
        ] * (len(p_tokens)) + [0] * len(tokens)
        re_sample = {
            'tokens': tokens,
            'token_ids': token_ids,
            'token_types': token_types,
            'label': label
        }
        return re_sample

    def _get_tag_list(self, tokens, span_tokens, tag_list):
        tag_start_index = tokens.index(span_tokens[0])
        tag_end_index = min((tag_start_index + len(span_tokens)),
                            len(tag_list))
        for i in range(tag_start_index, tag_end_index):
            if tag_start_index == i:
                tag_list[i] = self.tag_to_ix["B"]
            else:
                tag_list[i] = self.tag_to_ix["I"]
        return tag_list

    def _padding(self, data, max_len, is_label=False):
        res = []
        for seq in data:
            if len(seq) > max_len:
                res.append(seq[:max_len])
            else:
                if is_label:
                    res.append(seq + [-1] * (max_len - len(seq)))
                else:
                    res.append(seq + [0] * (max_len - len(seq)))
        return res


if __name__ == "__main__":
    # test the data
    dl = BatchLoader(args)
    data = dl.load_data('smallTrain.json')
    ner_data, re_data = dl.build_data(data, True)
    bgs = dl.batch_loader(ner_data, re_data, is_train=True, batch_size=1)
    # Multi-task learning with task_id
    task_ids = [0] * 10 + [1] * 10
    shuffle(task_ids)
    print(task_ids)
    # 查看
    print(len(bgs[1]))
    print(bgs[1])
    iters = [iter(bg) for bg in bgs]
    for i in range(len(bgs[0])):
        sample = ner_data[i]
        print(sample)
        print('len of token: {}'.format(len(sample['tokens'])))
        print('len of token_ids: {}'.format(len(sample['token_ids'])))
        print('len of token_types: {}'.format(len(sample['token_types'])))
        print('len of tags: {}'.format(len(sample['tags'])))
        print('len of input_mask: {}'.format(len(sample['input_mask'])))
        print('len of output_mask: {}'.format(len(sample['output_mask'])))
        print(next(iters[0]))
        print('****')
        input()
