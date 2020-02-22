#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import logging
import os
import pickle
import random
import time
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from batch_loader import BatchLoader
from bert.optimization import BertAdam
from config.config import args
from model.modelJoint import KBQA
from util import utils

if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)


def eval(model,
         ner_dev_data,
         dev_data,
         dev_bl,
         graph,
         entity_linking,
         args,
         epoch=-1,
         evalwriter=None,
         errorsOut=False):
    ## Evaluate
    model.eval()
    dev_iter = iter(dev_bl)
    num = cor_num_s = cor_num_p = cor_num_o = 0
    t = trange(len(dev_bl), desc='Eval')
    errors = []
    for i in t:
        batch_data = next(dev_iter)
        batch_samples = dev_data[args.batch_size * i:args.batch_size * i +
                                 batch_data[0].size(0)]
        batch_ners = ner_dev_data[args.batch_size * i:args.batch_size * i +
                                  batch_data[0].size(0)]
        batch_data = tuple(tmp.to(args.device) for tmp in batch_data)
        taglist = model(batch_data, 0, False).tolist()
        for j in range(len(taglist)):
            tokens = batch_ners[j]['tokens']
            tags = [i for i in taglist[j] if i != -1]
            # get subject
            subject = utils.extract_sub(tags, tokens)
            if subject:
                subject = ''.join(subject).replace('##', '').replace('□', ' ')
                # get question and gold_spo
                question = batch_samples[j]['question']
                gold_spo = batch_samples[j]['triple']
                # 去掉书名号
                if subject[0] == '《':
                    subject = subject[1:]
                if subject[-1] == '》':
                    subject = subject[:-1]
                # 英文小写，删除[]以及里面的内容，数字规范化
                subject = utils.str2num(utils.symbolPro(subject.lower()))
                spos = graph.get(subject, [])
                ons = entity_linking.get(subject, [])
                for on in ons:
                    spos += graph.get(on, [])
                pres = list(set([spo[1] for spo in spos]))
                objs = set()
                pre = ''
                if pres:
                    sub_re_data = bl.build_re_data(question, pres)
                    sub_re_bl = bl.batch_loader(None,
                                                sub_re_data,
                                                args.ner_max_len,
                                                args.re_max_len,
                                                args.batch_size,
                                                is_train=False)
                    sub_labels = []
                    for batch_data in sub_re_bl:
                        batch_data = tuple(
                            tmp.to(args.device) for tmp in batch_data)
                        label_logits = model(batch_data, 1,
                                             False).cpu().tolist()
                        sub_labels += label_logits
                    index_pre = np.argmax(sub_labels)
                    pre = pres[index_pre].replace(' ', '')
                    pre = utils.str2num(utils.symbolPro(pre.lower()))
                    for spo in spos:
                        s, p, o = spo
                        if subject in s and p == pre:
                            objs.add(o)
                # 针对时间类问题，统一转化为2020-2-15的形式
                if utils.is_date(gold_spo[-1]):
                    answer = utils.str2date(gold_spo[-1])
                    objs = [
                        utils.str2date(obj) for obj in objs
                        if utils.is_date(obj)
                    ]
                # 数字范围
                # elif utils.is_number(gold_spo[-1]):
                #     number = filter(str.isdigit, gold_spo[-1])
                #     objs = [
                #         filter(str.isdigit, obj) for obj in objs
                #         if utils.is_number(obj)
                #     ]
                else:
                    answer = gold_spo[-1]               
                # 书名号
                if gold_spo[0][0] == '《':
                    subject = "《"+subject+"》"
                num += 1
                cor_num_s += 1 if subject == gold_spo[0] else 0
                cor_num_p += 1 if len(utils.fuzzy_search([pre],
                                                         gold_spo[1])) > 0 else 0
                cor_num_o += 1 if len(utils.fuzzy_search(objs,
                                                         answer)) > 0 else 0
            else:
                num += 1
                cor_num_s += 0
                cor_num_p += 0
                cor_num_o += 0

            if errorsOut:
                sample = {'question': question, 'triple': gold_spo}
                if subject != gold_spo[0]:
                    sample['subject'] = subject
                    sample['taglist'] = tags
                if pre != gold_spo[1]:
                    sample['pre'] = pre
                if gold_spo[-1] not in objs:
                    sample['objs'] = list(objs)
                if len(sample) > 2:
                    errors.append(sample)
            # acc_s 是实体准确率，acc_p是关系准确率，acc_o
            t.set_postfix(acc_s='{:.2f}'.format(cor_num_s / num * 100),
                          acc_p='{:.2f}'.format(cor_num_p / num * 100),
                          acc_o='{:.2f}'.format(cor_num_o / num * 100))

        if evalwriter and epoch > -1:
            evalwriter.add_scalars(
                'data', {
                    'acc_s': cor_num_s / num * 100,
                    'acc_p': cor_num_p / num * 100,
                    'acc_o': cor_num_o / num * 100
                }, epoch * i)

        logging.info("acc_s={:.2f},acc_p={:.2f},acc_o={:.2f}".format(
            cor_num_s / num * 100, cor_num_p / num * 100,
            cor_num_o / num * 100))

        if errors:
            with open('error/error.json', 'w', encoding='utf-8') as f:
                f.write(
                    json.dumps(errors,
                               ensure_ascii=False,
                               cls=utils.MyEncoder,
                               indent=len(errors[0])) + '\n')

    return cor_num_o / num


if __name__ == '__main__':
    # mark time
    current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())

    # Use GPUs if available
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # CRF need use_cuda
    args.use_cuda = False
    # load model
    args.cuda = True if torch.cuda.is_available() else False

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info('device: {}'.format(args.device))
    logging.info('Hyper params:%r' % args.__dict__)

    # Create the input data pipeline
    logging.info('Loading the datasets...')
    bl = BatchLoader(args)

    # # Load train and dev data （for debug）
    # train_data = bl.load_data('smallTrain.json')
    # dev_data = bl.load_data('smallDev.json')

    # Load train and dev data
    train_data = bl.load_data('train.json')
    dev_data = bl.load_data('dev.json')

    ## Train data
    ner_train_data, re_train_data = bl.build_data(train_data, is_train=True)
    train_bls = bl.batch_loader(ner_train_data,
                                re_train_data,
                                args.ner_max_len,
                                args.re_max_len,
                                args.batch_size,
                                is_train=True)
    num_batchs_per_task = [len(train_bl) for train_bl in train_bls]
    logging.info(
        'num of batch per task for train: {}'.format(num_batchs_per_task))
    train_task_ids = sum([[i] * num_batchs_per_task[i]
                          for i in range(len(num_batchs_per_task))], [])
    shuffle(train_task_ids)

    ## Dev data
    ner_dev_data, _ = bl.build_data(dev_data, is_train=False)
    dev_bl = bl.batch_loader(ner_dev_data,
                             None,
                             args.ner_max_len,
                             args.re_max_len,
                             args.batch_size,
                             is_train=False)
    logging.info('num of batch for dev: {}'.format(len(dev_bl)))

    # Model
    model = KBQA.from_pretrained(args.bert_model_dir,
                                 num_tag=len(args.tag_to_ix),
                                 use_cuda=args.use_cuda)
    model.to(args.device)
    logging.info('total parameters:{}'.format(utils.model_scale(model)))

    # writer = SummaryWriter('visualization/{}/{}'.format(current_time, 'model'))
    trainwriter = SummaryWriter('visualization/{}/{}'.format(
        current_time, 'Train'))
    evalwriter = SummaryWriter('visualization/{}/{}'.format(
        current_time, 'Eval'))

    # Optimizer
    param_optimizer = list(model.named_parameters())

    no_decay = [
        'bias', 'LayerNorm.bias', 'LayerNorm.weight', 'crf.transitions'
    ]

    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'names':
        [n for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate':
        0.01,
    }, {
        'params': [
            p for n, p in param_optimizer
            if any(nd in n for nd in no_decay[:-1])
        ],
        'names': [
            n for n, p in param_optimizer
            if any(nd in n for nd in no_decay[:-1])
        ],
        'weight_decay_rate':
        0.0,
    }, {
        'params': [p for n, p in param_optimizer if n == no_decay[-1]],
        'names': [n for n, p in param_optimizer if n == no_decay[-1]],
        'weight_decay_rate':
        0.0,
        'lr':
        args.learning_rate * 100
    }]

    args.steps_per_epoch = sum(num_batchs_per_task)
    args.total_steps = args.steps_per_epoch * args.epoch_num
    optimizer = BertAdam(params=optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup,
                         t_total=args.total_steps,
                         max_grad_norm=args.clip_grad,
                         schedule=args.schedule)

    logging.info('Loading graph and entity linking...')
    graph = pickle.load(open('data/graph/graph.pkl', 'rb'))
    entity_linking = pickle.load(open('data/graph/entity_linking.pkl', 'rb'))

    if args.do_train_and_eval:
        # reload saved model
        if os.path.exists(os.path.join(args.model_dir, 'last.pth.tar')):
            _, start_epoch = utils.load_checkpoint(os.path.join(
                args.model_dir, 'last.pth.tar'),
                                                   model,
                                                   cuda=args.cuda)
            logging.info('Loaded epoch {} successfully'.format(start_epoch))
            start_epoch += 1
        else:
            start_epoch = 0
            logging.info('No-save model, training from begin')

        # Train and evaluate
        best_acc = 0
        for epoch in range(start_epoch, args.epoch_num):
            logging.info(" Training epoch: {}".format(epoch))
            ## Train
            model.train()
            # 进度条
            t = trange(args.steps_per_epoch,
                       desc='Epoch {} -Train'.format(epoch))
            loss_avg = utils.RunningAverage()
            train_iters = [iter(tmp) for tmp in train_bls
                           ]  # to use next and reset the iterator
            for i in t:
                task_id = train_task_ids[i]
                batch_data = next(train_iters[task_id])
                batch_data = tuple(tmp.to(args.device) for tmp in batch_data)
                loss = model(batch_data, task_id, True)
                # Generate model flowchart
                # if epoch == start_epoch:
                #     writer.add_graph(model, (batch_data, task_id, True))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_avg.update(loss.item())
                t.set_postfix(loss='{:5.4f}'.format(loss.item()),
                              avg_loss='{:5.4f}'.format(loss_avg()))

                if task_id == 0:
                    trainwriter.add_scalars('sub', {
                        'loss': loss.item(),
                        'avg_loss': loss_avg()
                    }, epoch * args.steps_per_epoch + i)
                else:
                    trainwriter.add_scalars('pre', {
                        'loss': loss.item(),
                        'avg_loss': loss_avg()
                    }, epoch * args.steps_per_epoch + i)

            if epoch % args.verbose == 0:
                logging.info("loss={:5.4f}, avg_loss={:5.4f}'".format(
                    loss.item(), loss_avg()))

            errorsOut = True if epoch >= (args.epoch_num -
                                          args.errors_verbose) else False

            acc = eval(model, ner_dev_data, dev_data, dev_bl, graph,
                       entity_linking, args, epoch, evalwriter, errorsOut)
            utils.save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict()
                },
                is_best=acc > best_acc,
                checkpoint=args.model_dir)
            best_acc = max(best_acc, acc)

    if args.do_eval:
        logging.info('num of batch for dev: {}'.format(len(dev_bl)))
        _, start_epoch = utils.load_checkpoint(os.path.join(
            args.model_dir, 'best.pth.tar'),
                                               model,
                                               cuda=args.cuda)
        logging.info('Loaded epoch {} successfully'.format(start_epoch))
        eval(model,
             ner_dev_data,
             dev_data,
             dev_bl,
             graph,
             entity_linking,
             args,
             errorsOut=True)

    if args.do_predict:
        utils.load_checkpoint(os.path.join(args.model_dir, 'best.pth.tar'),
                              model,
                              cuda=args.cuda)
        model.eval()
        logging.info('Loading graph and entity linking...')
        graph = pickle.load(open('data/graph/graph.pkl', 'rb'))
        entity_linking = pickle.load(
            open('data/graph/entity_linking.pkl', 'rb'))
        while True:
            try:
                logging.info('Enter a question:')
                question = input()
                ner_data = bl.build_ner_data(question)
                ner_bl = bl.batch_loader(ner_data,
                                         None,
                                         args.ner_max_len,
                                         args.re_max_len,
                                         args.batch_size,
                                         is_train=False)
                for batch_data in ner_bl:
                    batch_data = tuple(
                        tmp.to(args.device) for tmp in batch_data)
                    taglist = model(batch_data, 0, False).tolist()
                    tags = [i for i in taglist[0] if i != -1]
                    tokens = ner_data[0]['tokens']
                    subject = utils.extract_sub(tokens, tags)
                    if subject:
                        subject = ''.join(subject).replace('##', '').replace(
                            '□', ' ')

                        logging.info('The subject is：{}'.format(subject))
                        if subject[0] == '《':
                            subject = subject[1:]
                        if subject[-1] == '》':
                            subject = subject[:-1]
                        subject = utils.str2num(
                            utils.symbolPro(subject.lower()))
                        spos = []
                        spos += graph.get(subject, [])
                        ons = entity_linking.get(subject, [])
                        for on in ons:
                            spos += graph.get(on, [])
                        spos = set(spos)
                        pres = list(set([spo[1] for spo in spos]))
                        # logging.info('Candidate relationship is：{}'.format(pres))
                        if pres:
                            sub_re_data = bl.build_re_data(question, pres)
                            sub_re_bl = bl.batch_loader(None,
                                                        sub_re_data,
                                                        args.ner_max_len,
                                                        args.re_max_len,
                                                        args.batch_size,
                                                        is_train=False)
                            sub_labels = []
                            for batch_data in sub_re_bl:
                                batch_data = tuple(
                                    tmp.to(args.device) for tmp in batch_data)
                                label_logits = model(batch_data, 1,
                                                     False).cpu().tolist()
                                sub_labels += label_logits
                            index_pre = np.argmax(sub_labels)
                            pre = pres[index_pre].replace(' ', '')
                            pre = utils.str2num(utils.symbolPro(pre.lower()))
                            logging.info(
                                'The most likely relationship is: {}'.format(
                                    pre))
                            for spo in spos:
                                s, p, o = spo
                                if subject in s and p == pre:
                                    logging.info('Answer：{}'.format(spo))
                    else:
                        logging.info('ner error!')
                logging.info('\n')
            except:
                logging.info('error! Whether to continue? y/n')
                cmd = input()
                if cmd == 'y':
                    pass
                else:
                    break
