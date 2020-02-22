###
 # @Author: Hong Jing Li
 # @Date: 2020-02-05 15:29:31
 # @LastEditors  : Hong Jing Li
 # @LastEditTime : 2020-02-11 14:50:54
 # @Contact: lihongjing.more@gmail.com
 ###

# 1. download NLPCC2016KBQA
echo "Download Q & A dataset"
git clone https://github.com/huangxiangzhou/NLPCC2016KBQA.git
mv nlpcc-iccpol-2016.kbqa.testing-data nlpcc-iccpol-2016.kbqa.training-data -t data/QAdata
rm -rf NLPCC2016KBQA

# 2. Generate dataset
echo "Generate Q & A dataset"
python data/data_generation/nega_sampling.py

# 3. Generate small dataset
cd data/QAdata
head -n 250 train.json > smallTrain.json
head -n 50 dev.json > smallDev.json

# 4. Organize the data according to the tree diagram given by readme
cd ..
if [ ! -d "data/NLPCC2017-OpenDomainQA" ]; then
  echo "You need to download NLPCC2017/nlpcc-iccpol-2016.kbqa.kb and nlpcc-iccpol-2016.kbqa.kb.mention2id to the data folder first, https://pan.baidu.com/s/1dEYcQXz"
else
  echo "Generate graph data"
  python data_generation/graph.py

# 5. Model weight
cd ..
cd model/model_params
wegt https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip 
unzip chinese_L-12_H-768_A-12.zip
mv chinese_L-12_H-768_A-12 chinese_wwm_ext_pytorch

echo "done"