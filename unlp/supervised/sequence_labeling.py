# -*- coding: utf-8 -*-

"""
@Time    : 2022/2/24 10:30 上午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""

import os
import time
file_root = os.path.dirname(__file__)
import sys
sys.path.append(file_root)
from ner.run import Service

class NER(object):
    def __init__(self, model_name, mode='predict', **kwargs):
        model_types = ['bert-base-chinese', "chinese-bert-wwm-ext", "ernie-1.0", "albert-base-chinese"]
        assert mode in ['predict', 'train', 'evaluate'], 'mode should in {}'.format(['predict', 'train', 'evaluate'])

        if model_name in model_types:
            self.model = Service(model_name, mode, **kwargs)
        else:
            print("suport model_type: {}".format("##".join(model_types)))
            os._exit(-1)

        self.mode = mode
        self.kwargs = kwargs

    def run(self, text=[]):
        b0 = time.time()
        if self.mode == 'train':
            res = self.model.run_train()
        elif self.mode == 'evaluate':
            res = self.model.run_evaluate()
        else:
            res = self.model.run_predict(text)
        print('cost {}'.format(time.time() - b0))
        return res

if __name__ == '__main__':
    # 预训练时可以通过resume (bool类型）控制是否继续训练，其他predict和evaluate阶段可以不传入这个参数
    ner = NER("bert-base-chinese", mode='predict', **{"data_dir": '/Volumes/work/project/unlp/unlp/supervised/ner/data/cluner',
                                        "model_path":"/Volumes/work/project/unlp/unlp/supervised/ner/data/cluner/saved_dict/bert-base-chinese"})
    text = ['在豪门被多线作战拖累时，正是他们悄悄追赶上来的大好时机。重新找回全队的凝聚力是拉科赢球的资本。',"不久后，“星展中国”南宁分行也将择机开业。除新加坡星展银行外，多家外资银行最近纷纷扩大在华投资，"]
    # print(res)
    res = ner.run(text=text)
    print(res)