# -*- coding: utf-8 -*-

"""
@Time    : 2022/3/7 5:13 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
from unlp import DataAugment

def dataaugment(mode, model_path, num_aug, text:str):
    # model type必须在model types内
    model_types = ['BERT', "BART", "EDA", "Translate"]
    if mode not in model_types:
        print("suport model_type: {}".format("##".join(model_types)))
        os._exit(-1)
    if model_path == '' and mode in ['BERT', "BART"]:
        print('model download automatically!')
    model = DataAugment(mode=mode, model_path=model_path, num_aug=num_aug)

    res = model.run(text=text)
    return res

if __name__ == '__main__':
    model_path = '/data/lss/deepenv/deepenv-data/unlp包/transformer/word2vec/light_Tencent_AILab_ChineseEmbedding.bin'
    model_path = '/data/lss/deepenv/deepenv-data/unlp包/transformer/bart-base-chinese'
    res = dataaugment(model_path=model_path, mode='BART', num_aug=15, text='哪款可以保留矿物质')
    print(res)