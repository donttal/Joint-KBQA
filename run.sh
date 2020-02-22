###
 # @Author: Hong Jing Li
 # @Date: 2020-01-12 22:37:46
 # @LastEditors: Hong Jing Li
 # @LastEditTime : 2020-02-11 12:46:00
 # @Contact: lihongjing.more@gmail.com
 ###
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=1

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