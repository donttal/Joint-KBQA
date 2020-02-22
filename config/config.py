'''
@Author: Hong Jing Li
@Date: 2020-01-12 18:37:19
@LastEditors: Hong Jing Li
@LastEditTime : 2020-02-11 00:25:38
@Contact: lihongjing.more@gmail.com
'''
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# import lib
import argparse
import os

import torch

parser = argparse.ArgumentParser()

# file path
parser.add_argument('--data_dir',
                    default='data/QAdata',
                    help="Directory containing the dataset")
parser.add_argument('--bert_model_dir',
                    default='model/model_params/chinese_wwm_ext_pytorch',
                    help="Directory containing the BERT model in PyTorch")
parser.add_argument('--model_dir',
                    default='output/baseline',
                    help="model directory")

# Hyperparameter
parser.add_argument('--clip_grad', default=2, type=int, help="")

parser.add_argument('--seed',
                    default=42,
                    type=int,
                    help="random seed for initialization")

parser.add_argument('--schedule',
                    default='warmup_linear',
                    help="schedule for optimizer")

parser.add_argument('--weight_decay', default=0.01, type=float, help="")

parser.add_argument('--warmup', default=0.1, type=float, help="")

parser.add_argument('--epoch_num', default=8, type=int, help="num of epoch")

parser.add_argument('--nega_num',
                    default=4,
                    type=int,
                    help="num of negative predicates")

parser.add_argument('--batch_size', default=64, type=int, help="batch size")

parser.add_argument('--ner_max_len',
                    default=64,
                    type=int,
                    help="max sequence length for ner task")

parser.add_argument('--re_max_len',
                    default=64,
                    type=int,
                    help="max sequence length for re task")

parser.add_argument('--learning_rate',
                    default=5e-5,
                    type=float,
                    help="learning rate")

# Model operation mode
parser.add_argument('--do_train_and_eval',
                    action='store_true',
                    help="do_train_and_eval")
parser.add_argument('--do_eval', action='store_true', help="do_eval")
parser.add_argument('--do_predict', action='store_true', help="do_predict")

# Data logging parameters
parser.add_argument('--verbose',
                    default=2,
                    help="Save parameters when epoch%verbose==0")

parser.add_argument('--errors_verbose',
                    default=1,
                    help="extract errors from del")

# tags
tag_to_ix = {"B": 0, "I": 1, "O": 2}
ix_to_tag = {0: 'B', 1: 'I', 2: 'O'}
parser.add_argument('--tag_to_ix', default=tag_to_ix, help="ner tag list")
parser.add_argument('--ix_to_tag', default=ix_to_tag, help="ner convert list")

args = parser.parse_args()
