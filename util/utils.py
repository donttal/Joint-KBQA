import difflib
import json
import logging
import os
import re
import shutil

import numpy as np
import torch
from tqdm import tqdm


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + \
                    self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print(
            "Checkpoint Directory does not exist! Making directory {}".format(
                checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None, cuda=True):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    logging.info('Loading the ckpt from {}'.format(checkpoint))
    checkpoint = torch.load(
        checkpoint,
        map_location=torch.device('cpu')) if cuda else torch.load(checkpoint)

    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    # last epochs
    start_epoch = checkpoint['epoch']

    return checkpoint, start_epoch


def fuzzy_search(sourceList, word, cutoff=0.5):
    return difflib.get_close_matches(word, sourceList, cutoff=cutoff)


def extract_sub(tags, tokens):
    subjects = []
    tmp = []
    for i, tag in enumerate(tags):
        if tag == 0:
            if tmp:
                subjects.append(tmp)
                tmp = []
                tmp.append(tokens[i + 1])
            else:
                tmp.append(tokens[i + 1])
        if tag == 1 and tmp:
            tmp.append(tokens[i + 1])
        if (tag == 2 or i == len(tags)) and tmp:
            subjects.append(tmp)
            tmp = []
    if subjects == []:
        tmp = []
        for i, tag in enumerate(tags):
            if tag == 1:
                tmp.append(tokens[i + 1])
            if (tag == 2 or i == len(tags)) and tmp:
                subjects.append(tmp)
                tmp = []
    return subjects[0] if subjects else []


def model_scale(model):
    # See the size of model parameters
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    return k


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass


def is_date(str_data):
    s = str_data.replace("年",
                         "-").replace("月",
                                      "-").replace("日",
                                                   " ").replace("/",
                                                                "-").strip()
    pattern = re.compile(u'\d{4}\-\d{1,2}\-\d{1,2}')
    return True if pattern.search(s) else False


def str2num(s):
    if is_number(s):
        return s.rstrip('0').strip('.') if '.' in s else s
    else:
        return s


def str2date(str_date):
    str_date = str_date.strip()
    year = 1900
    month = 1
    day = 1
    if (len(str_date) > 11):
        str_date = str_date[:11]
    if (str_date.find('-') > 0):
        year = str_date[:4]
        if (year.isdigit()):
            year = int(year)
        else:
            year = 0
        month = str_date[5:str_date.rfind('-')]
        if (month.isdigit()):
            month = int(month)
        else:
            month = 0
        if (str_date.find(' ') == -1):
            day = str_date[str_date.rfind('-') + 1:]
        else:
            day = str_date[str_date.rfind('-') + 1:str_date.find(' ')]
        if (day.isdigit()):
            day = int(day)
        else:
            day = 0
    elif (str_date.find('年') > 0):
        year = str_date[:4]
        if (year.isdigit()):
            year = int(year)
        else:
            year = 0
        month = str_date[5:str_date.rfind('月')]
        if (month.isdigit()):
            month = int(month)
        else:
            month = 0
        day = str_date[str_date.rfind('月') + 1:str_date.rfind('日')]
        if (day.isdigit()):
            day = int(day)
        else:
            day = 0
    elif (str_date.find('/') > 0):
        year = str_date[:4]
        if (year.isdigit()):
            year = int(year)
        else:
            year = 0
        month = str_date[5:str_date.rfind('/')]
        if (month.isdigit()):
            month = int(month)
        else:
            month = 0
        if (str_date.find(' ') == -1):
            day = str_date[str_date.rfind('/') + 1:]
        else:
            day = str_date[str_date.rfind('/') + 1:str_date.find(' ')]
        if (day.isdigit()):
            day = int(day)
        else:
            day = 0
    else:
        year = 1900
        month = 1
        day = 1
    if month < 10:
        month = '0' + str(month)
    if day < 10:
        day = '0' + str(day)
    return '%s-%s-%s' % (year, month, day)


def symbolPro(s):
    return re.sub(u"\\[.*?\\]", "", s)


if __name__ == "__main__":
    date = '1964年4月20日'
    objs = ['1964-04-20']
    if is_date(date):
        answer = str2date(date)
        objs = [str2date(obj) for obj in objs if is_date(obj)]
        print(fuzzy_search(objs,answer))

    # num = '约39.99km/h'
    # objs = ['39.990000']
    # n = str2num(re.findall(r'\d+\.\d*', num)[0])
    # if is_number(n):
    #     objs = [str2num(re.findall(r'\d+\.\d*', obj)[0]) for obj in objs]
    #     print(objs, n)

    # objs = ["0852","0735","0564","0746","0775","0872","0554","020","0797","0595"]
    # answer = '746'
    # print(len(fuzzy_search(objs, answer)))
