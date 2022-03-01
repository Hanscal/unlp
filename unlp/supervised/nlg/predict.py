# -*- coding: utf-8 -*-
"""
预处理：构建一个Batch对象
    1 首先得到article的分词
    构建一个example Example(art_str, ["",], vocab)
        art_str 是分词后用空格连起来的字符串
        Example的第二个参数是abstract的分词空格连接在一起的各个句子列表[句子1，句子2.。。]
    2 构建一个Batch
    [ex1,ex2...] + vocab + bach_size 构建一个Batch
"""
import jieba
import os
import torch
import sys
file_root = os.path.dirname(__file__)
sys.path.append(file_root)

from gutils.batcher import Example
from gutils.batcher import Batch
from gmodels.beam_search_decode import BeamSearch
from gutils.config import get_argparse


class Predictor(object):
    def __init__(self, **kwargs):
        config = get_argparse()
        args_bak = vars(config)
        for k, v in kwargs.items():
            if k in args_bak:
                args_bak[k] = v

        # set cuda and device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args_bak['device'] = device
        config.batch_size = config.beam_size  # 这个很重要
        self.config = config
        self.beam_processor = BeamSearch(config=config)

    def build_batch_by_article(self, article, vocab, beam_size):
        words = jieba.cut(article)
        art_str = " ".join(words)
        example = Example(art_str, ["",], vocab)
        ex_list = [example for _ in range(beam_size)]
        batch = Batch(ex_list, vocab, self.config)

        return batch

    def run_predict(self, texts:list):
        res = []
        for text in texts:
            batch = self.build_batch_by_article(text, self.beam_processor.vocab, self.config.beam_size)
            summary = self.beam_processor.decode(batch)
            res.append(summary)
        return res


if __name__ == '__main__':
    article = "近日，一段消防员用叉子吃饭的视频在网上引起热议。原来是因为训练强度太大，半天下来，大家拿筷子的手一直在抖，甚至没法夹菜。于是，用叉子吃饭，渐渐成了上海黄浦消防车站中队饭桌上的传统。转发，向消防员致敬！"
    # model_path = sys.argv[1]
    model_path = os.path.join(file_root, 'data/weibo/saved_dict/point-net/point-net.pt')
    p = Predictor(**{"model_path":model_path,"data_dir":os.path.join(file_root,'data/weibo')})
    res = p.run_predict([article])
    print(res)