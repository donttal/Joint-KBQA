'''
@Author: Hong Jing Li
@Date: 2020-01-13 12:40:40
@LastEditors  : Hong Jing Li
@LastEditTime : 2020-02-11 14:51:35
@Contact: lihongjing.more@gmail.com
'''
# coding=utf-8
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import binary_cross_entropy_with_logits
from tqdm import tqdm

from batch_loader import WHITESPACE_PLACEHOLDER
from bert import BertModel, BertPreTrainedModel

from .crf import CRF


class KBQA(BertPreTrainedModel):
    def __init__(self, config, num_tag, use_cuda):
        super(KBQA, self).__init__(config)
        # BERT
        self.bert = BertModel(config)

        # NER
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_tag)
        self.crf = CRF(num_tag, use_cuda)

        # relationship
        self.re_layer = nn.Linear(config.hidden_size, 1)  # yes/no
        self.apply(self.init_bert_weights)

    def forward(self, batch_data, task_id, is_train):
        if task_id == 0:  # task is ner
            token_ids, token_types, tag_list, input_mask, output_mask = batch_data
            bert_encode, _ = self.bert(token_ids,
                                       token_types,
                                       input_mask,
                                       output_all_encoded_layers=False)

            output = self.classifier(bert_encode)
            logits = self.crf.get_batch_best_path(output, output_mask)
            if is_train:
                loss = self.crf.negative_log_loss(output, output_mask,
                                                  tag_list)

        else:  # task is re
            token_ids, token_types, label = batch_data
            attention_mask = token_ids.gt(0)
            _, pooled_output = self.bert(token_ids,
                                         token_types,
                                         attention_mask,
                                         output_all_encoded_layers=False)
            # add another layer
            logits = self.re_layer(pooled_output).squeeze(-1)
            if is_train:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, label)
        return loss if is_train else logits
