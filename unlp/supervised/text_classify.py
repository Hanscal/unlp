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
from classification.run import Service

class Classification(object):
    def __init__(self, model_name, mode='predict', use_word=False, **kwargs):
        model_types = ['DPCNN', "FastText", "TextCNN", "TextRCNN", "TextRNN", "TextRNN_Att", "BERT", "ERNIE"]
        assert mode in ['predict', 'train', 'evaluate'], 'mode should in {}'.format(['predict', 'train', 'evaluate'])

        if model_name in model_types:
            self.model = Service(model_name, mode, use_word=use_word, **kwargs)
        else:
            print("suport model_type: {}".format("##".join(model_types)))
            os._exit(-1)

        self.mode = mode
        self.kwargs = kwargs

    def run(self):
        b0 = time.time()
        if self.mode == 'train':
            res = self.model.run_train()
        elif self.mode == 'evaluate':
            res = self.model.run_evaluate()
        else:
            text = self.kwargs.pop('text')
            res = self.model.run_predict(text)
        print('cost {}'.format(time.time() - b0))
        return res

if __name__ == '__main__':
    classify = Classification("BERT", mode='predict', use_word=False, **{"embedding":"random", "dataset": '/Volumes/work/project/unlp/unlp/supervised/classification/data/THUCNews',
                                                                         "pretrain_model_path":"/Volumes/work/project/unlp/unlp/transformers/bert-base-chinese",
                                                                          "model_path":"/Volumes/work/project/unlp/unlp/supervised/classification/data/THUCNews/saved_dict/BERT.ckpt",
                                                                          "text":['艺龙网并购两家旅游网站',"封基上周溃退 未有明显估值优势","中华女子学院：本科层次仅1专业招男生"]})
    res = classify.run()
    print(res)