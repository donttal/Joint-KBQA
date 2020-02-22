# 基于多任务学习BERT的KBQA

一个基于知识图谱的端到端问答系统，在BERT基础上做多任务联合训练。
An end-to-end question answering system based on knowledge graph, doing multi-task joint training on the basis of BERT.

**完整树状图**

```
├── README.md
├── batch_loader.py
├── bert
│   ├── __init__.py
│   ├── file_utils.py
│   ├── modeling.py
│   ├── optimization.py
│   └── tokenization.py
├── config
│   └── config.py
├── data
│   ├── NLPCC2017-OpenDomainQA
│   │   └── knowledge
│   │       ├── nlpcc-iccpol-2016.kbqa.kb
│   │       └── nlpcc-iccpol-2016.kbqa.kb.mention2id
│   ├── QAdata
│   │   ├── dev.json
│   │   ├── nlpcc-iccpol-2016.kbqa.testing-data
│   │   ├── nlpcc-iccpol-2016.kbqa.training-data
│   │   ├── smallDev.json
│   │   ├── smallTrain.json
│   │   └── train.json
│   ├── data_generation
│   │   ├── graph.py
│   │   └── nega_sampling.py
│   └── graph
│       ├── entity_linking.pkl
│       └── graph.pkl
├── data_setup.sh
├── error
│   ├── error.json
├── img
│   ├── example.png
│   └── model.png
├── main.py
├── model
│   ├── crf.py
│   ├── modelJoint.py
│   └── model_params
│       └── chinese_wwm_ext_pytorch
│           ├── bert_config.json
│           ├── pytorch_model.bin
│           └── vocab.txt
├── output
│   └── baseline
│       ├── best.pth.tar
│       ├── last.pth.tar
│       └── train.log
├── requirements.txt
├── run.sh
├── util
│   └── utils.py
└── visualization
    ├── 2020-02-12T10:29
    │   ├── Eval
    │   │   └── events.out.tfevents.1581474612.localhost.localdomain
    │   └── Train
    │       ├── events.out.tfevents.1581474612.localhost.localdomain
    │       └── pre
    │           ├── avg_loss
    │           │   └── events.out.tfevents.1581474673.localhost.localdomain
    │           └── loss
    │               └── events.out.tfevents.1581474673.localhost.localdomain
```

## setting
我们使用的数据集来自NLPCC ICCPOL 2016 KBQA 任务集，其包含 14 609 个问答对的训练集和包含 9 870 个问答对的测试集。 并提供一个知识库，包含 6 502 738 个实体、 587 875 个属性以及 43 063 796 个 三元组。知识库文件中每行存储一个事实 ，即三元组 ( 实体、属性、属性值) 。

我们使用google发表的预训练BERT-Chinese模型，该模型具有12层，768个隐藏状态，12个heads和110M个参数。 为了进行微调，所有超参数都将在开发集上进行调整。 根据我们的数据集，最大实体序列长度设置为64，最大谓词序列长度设置为64，批量大小设置为64。我们使用Adam进行优化，其中β1= 0.9和β2= 0.999。 dropout概率为0.1。 通常，对于BERT-CRF和BERT-Softmax，BERT的初始学习速率设置为5e-5，CRF和Softmax层的初始学习速率设置为5e-3。

## Introduction
本项目实现了一个基于BERT的端到端的KBQA系统，支持单跳问题的查询。主要分为两个部分：
1. 实体识别(NER)：输入一个问句，找出该问句的唯一核心实体(subject)；
2. 关系抽取(RE)：输入一个问句和一个关系，判断该问句是否询问了该关系(predicate)。

### **训练过程**
模型实体识别和关系抽取共用了一个BERT作为表达层。
1. 其中由于问题中确定只存在一个主语实体，NER部分就是一个简单的BERT+CRF的结构；
2. 关系抽取的部分，使用的是BERT+Softmax结构。在训练时涉及到负采样。方案如下：先根据gold triple中的主语，到图谱中找到该主语对应的所有关系(predicate)，剔除掉正确的关系后，剩下的关系作为负样本。正样本监督信号为1，负样本为0。

### **答案选择**
RE时，先用NER找到的主语，去图谱中找到其对应的所有关系，再逐个和问题配对，放到re模型中，选取置信度最高的关系，到图谱中寻找最后的问题答案(object)。

## Environment
```
Python3.6
PyTorch>=1.0
numpy==1.14.6
tqdm==4.31.1
```
详情可以查看项目的requirements.txt文件

## Usage
1. 运行`data_setup.sh`来构建项目数据集
2. 在`config.py`中配置超参；
3. 写`run.sh`，选择训练、验证还是测试模式。

run.sh例子
```
# # train
# python main.py \
#     --do_train_and_eval \
#     --model_dir output/baseline \
#     --nega_num 8 \
#     --learning_rate 5e-5 \
#     --batch_size 32 \
#     --epoch_num 16

# eval
python main.py \
    --do_eval

# # Predict
# python main.py \
#     --do_predict \
#     --model_dir output/baseline
```

## Reference

1. [NLPCC2016](https://github.com/huangxiangzhou/NLPCC2016KBQA)
2. [KBQA-BERT](https://github.com/WenRichard/KBQA-BERT)
3. [bert-kbqa-NLPCC2017](https://github.com/jkszw2014/bert-kbqa-NLPCC2017)
4. [Joint-BERT-KBQA](https://github.com/wangbq18/Joint-BERT-KBQA)
5. [BERT-NER-Pytorch](https://github.com/lonePatient/BERT-NER-Pytorch)
6. [Bert_Chinese_Ner_pytorch](https://github.com/circlePi/Bert_Chinese_Ner_pytorch)