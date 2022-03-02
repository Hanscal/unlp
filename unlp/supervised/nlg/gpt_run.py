# -*- coding: utf-8 -*-

"""
@Time    : 2022/2/24 11:48 上午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""

import os
import sys
import torch
import numpy as np
from typing import List

file_root = os.path.dirname(__file__)
sys.path.append(file_root)

from gpt_train import Train
from gpt_eval import Evaluate
from gpt_predict import Predictor

class Service(object):
    def __init__(self, model_name, mode, **kwargs):
        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样

        kwargs.update({'model_type':model_name})

        if mode == 'train':
            # model_path = kwargs['model_path']
            # data_dir = kwargs['data_dir'] 一定要包含data_dir
            self.trainer = Train(**kwargs)

        elif mode == 'evaluate':
            # load eval dataset
            # model_path = kwargs['model_path']  #一定要包含model_path 和 model_path
            # data_dir = kwargs['data_dir']
            self.evaluator = Evaluate(**kwargs)

        elif mode == 'predict':
            # model_path = os.path.join(file_root, 'data/weibo/saved_dict/point-net/point-net.pt') # 必须包含model_path
            self.predictor = Predictor(**kwargs)

        self.kwargs = kwargs

    def run_train(self):
        res = self.trainer.train()
        return res

    def run_evaluate(self, **kwargs):
        res = self.evaluator.run_eval()
        if "rouge" in kwargs:
            res = []
            for i in range(1,4):
                tmp = self.evaluator.run_rouge(kwargs['refs'], kwargs['preds'], i)
                res.append({"rouge_"+str(i):tmp})
        return res

    def run_predict(self, text: List[str]):
        res = self.predictor.run_predict(texts=text)
        return res


if __name__ == '__main__':
    article = "近日，一段消防员用叉子吃饭的视频在网上引起热议。原来是因为训练强度太大，半天下来，大家拿筷子的手一直在抖，甚至没法夹菜。于是，用叉子吃饭，渐渐成了上海黄浦消防车站中队饭桌上的传统。转发，向消防员致敬！"
    # model_path = sys.argv[1]
    model_path = os.path.join(file_root, 'data/weibo/saved_dict/point-net/point-net.pt')
    p = Service(**{"model_path": model_path, "data_dir": os.path.join(file_root, 'data/weibo')})
    res = p.run_predict([article])
    print(res)