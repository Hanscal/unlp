# -*- coding: utf-8 -*-

"""
@Time    : 2022/1/14 3:34 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
import time
file_root = os.path.dirname(__file__)
import sys
sys.path.append(file_root)
from nlg.run import Service

class Summarization(object):
    def __init__(self, model_name, mode='predict', **kwargs):
        self.model = Service(model_name, mode, **kwargs)
        self.mode = mode
        self.kwargs = kwargs

    def run(self,text=[]):
        b0 = time.time()
        if self.mode == 'train':
            res = self.model.run_train()
        elif self.mode == 'evaluate':
            res = self.model.run_evaluate(**self.kwargs)
        else:
            res = self.model.run_predict(text)
        print('cost {}'.format(time.time() - b0))
        return res

if __name__ == '__main__':
    # 预训练时可以通过resume (bool类型）控制是否继续训练，其他predict和evaluate阶段可以不传入这个参数
    summarize = Summarization("point-net", mode='predict', use_word=False, **{"data_dir": '/Volumes/work/project/unlp/unlp/supervised/nlg/data/weibo',
                                                                          "model_path":"/Volumes/work/project/unlp/unlp/supervised/nlg/data/weibo/saved_dict/point-net/point-net.pt"})
    res = summarize.run(text=['艺龙网并购两家旅游网站,封基上周溃退 未有明显估值优势,中华女子学院：本科层次仅1专业招男生'])
    print(res)