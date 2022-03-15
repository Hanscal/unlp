# -*- coding: utf-8 -*-

"""
@Time    : 2022/3/7 5:12 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
from unlp import SEntityRecognition

def entity_recognition(model_path, model_type, mode, datadir, text_list:list=[]):
    model_types = ['bert-base-chinese', "chinese-bert-wwm-ext", "ernie-1.0", "albert-base-chinese"]
    if model_type not in model_types:
        print("suport model_type: {}".format("##".join(model_types)))
        os._exit(-1)
    assert mode in ['predict', 'train', 'evaluate'], 'mode should in {}'.format(['predict', 'train', 'evaluate'])

    model = SEntityRecognition(model_path, model_type, mode, datadir)
    res = model.run(text=text_list)  # 实现模型的训练，评估和预测
    return res

if __name__ == '__main__':
    model_path = '/Volumes/work/project/unlp/unlp/transformers/bert-base-chinese'
    res = entity_recognition(model_path=model_path, model_type='bert-base-chinese', mode='train',datadir='/Volumes/work/project/unlp/unlp/supervised/ner/data/cluner')
    print(res)