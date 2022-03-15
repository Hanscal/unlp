# -*- coding: utf-8 -*-

"""
@Time    : 2022/3/7 5:11 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
from unlp import STextClassification

def classify(model_path, model_type, mode, datadir, text_list: list=[]):
    # model type必须在model types内
    model_types = ['DPCNN', "FastText", "TextCNN", "TextRCNN", "TextRNN", "TextRNN_Att", "BERT", "ERNIE"]
    if model_type not in model_types:
        print("suport model_type: {}".format("##".join(model_types)))
        os._exit(-1)
    model = STextClassification(model_path=model_path, model_type=model_type, mode=mode,datadir=datadir)
    res = model.run(text=text_list)
    return res

if __name__ == '__main__':
    model_path = '/Volumes/work/project/unlp/unlp/transformers/bert-base-chinese'
    res = classify(model_path=model_path, model_type='BERT', mode='train', datadir='/Volumes/work/project/unlp/unlp/supervised/classification/data/THUCNews')
    print(res)