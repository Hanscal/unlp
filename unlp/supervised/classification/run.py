# -*- coding: utf-8 -*-

"""
@Time    : 2022/2/14 5:59 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import json
import os
import sys
import torch
import numpy as np
from importlib import import_module
from typing import List
file_root = os.path.dirname(__file__)
sys.path.append(file_root)

from train import Train, TrainTransfomer
from evaluate import Evaluate
from predict import Predict

from sutils.dutils import init_network
from sutils.dutils import load_vocab

WEIGHTS_NAME = 'pytorch_model.bin'

class Service(object):
    def __init__(self, model_name, mode, use_word=False, **kwargs):
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        self.model_name = model_name

        if self.model_name in ['BERT', 'ERNIE']:
            self.x = import_module('unlp.supervised.classification.models.transformer.' + self.model_name)
        else:
            self.x = import_module('unlp.supervised.classification.models.' + self.model_name)

        # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
        if self.model_name == 'FastText':
            from sutils.dutils import build_fasttext_dataset as build_dataset
            from sutils.dutils import build_iterator
            self.embedding = 'random'
        elif self.model_name in ['BERT', "ERNIE"]:
            from sutils.dutils import build_tranformer_dataset as build_dataset
            from sutils.dutils import build_transformer_iterator as build_iterator
        else:
            from sutils.dutils import build_dataset, build_iterator

        self.build_dataset = build_dataset
        self.build_iterator = build_iterator
        self.use_word = use_word

        dataset = kwargs.pop('dataset')
        embedding = kwargs.pop('embedding')
        if mode == 'train':
            self.config = self.x.Config(dataset, embedding, **kwargs)
            self.vocab = load_vocab(self.config, self.use_word)
            self.config.n_vocab = len(self.vocab)
            self.model = self.x.Model(self.config).to(self.config.device)
            resume_model_path = kwargs.get('model_path', '')
            if os.path.isdir(resume_model_path) and kwargs.get('resume',''):
                resume_model_path = os.path.join(resume_model_path, WEIGHTS_NAME)
                print("resume model from {}".format(resume_model_path))
                self.model.load_state_dict(torch.load(resume_model_path, map_location=torch.device('cpu')))
            elif os.path.isfile(resume_model_path) and kwargs.get('resume',''):
                print("resume model from {}".format(resume_model_path))
                self.model.load_state_dict(torch.load(resume_model_path, map_location=torch.device('cpu')))

        elif mode == 'evaluate':
            self.config = self.x.Config(dataset, embedding, **kwargs)
            self.vocab = load_vocab(self.config, use_word=self.use_word)
            self.config.n_vocab = len(self.vocab)
            self.model = self.x.Model(self.config).to(self.config.device)
            model_path = kwargs['model_path']
            if os.path.isdir(model_path):
                model_path = os.path.join(model_path, WEIGHTS_NAME)
            print("loading model from {}".format(model_path))
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        elif mode == 'predict':
            self.config = self.x.Config(dataset, embedding, **kwargs)
            try:
                vocab_path = self.config.__getattribute__('vocab_path')
                self.vocab = json.load(open(vocab_path))
                self.config.n_vocab = len(self.vocab)
            except:
                print('no vocab path in config, model name:{}'.format(self.config.model_name))
            model_path = kwargs['model_path']
            self.model = self.x.Model(self.config).to(self.config.device)
            if os.path.isdir(model_path):
                model_path = os.path.join(model_path, WEIGHTS_NAME)
            print("loading model from {}".format(model_path))
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def run_train(self): # 或者是pre_trained的路径
        if self.model_name not in ['BERT', 'ERNIE']:
            init_network(self.model)
            self.trainer = Train()
        else:
            self.trainer = TrainTransfomer()

        train_data, dev_data, test_data = self.build_dataset(self.config, **{"use_word":self.use_word, "vocab":self.vocab})
        train_iter = self.build_iterator(train_data, self.config)
        dev_iter = self.build_iterator(dev_data, self.config)
        test_iter = self.build_iterator(test_data, self.config)
        self.trainer.train(self.config, self.model, train_iter, dev_iter, test_iter)


    def run_evaluate(self):
        train_data, dev_data, test_data = self.build_dataset(self.config, **{"use_word": self.use_word, "vocab": self.vocab})
        test_iter = self.build_iterator(test_data, self.config)

        self.evaluator = Evaluate()
        acc, loss, report, confusion = self.evaluator.test(self.config, self.model, test_iter)
        return acc, loss, report, confusion


    def run_predict(self, text: List[str]):
        predictor = Predict(config=self.config, use_word=self.use_word)
        res = predictor.predict(model=self.model, texts=text)
        res_final = [self.config.class_list[i] for i in res]
        return res_final


if __name__ == '__main__':
    # train-DPCNN
    # service = Service(model_name="DPCNN", mode='train', use_word=True, **{"embedding":"random", "dataset": '/Volumes/work/project/unlp/unlp/supervised/classification/data/THUCNews',
    #                                                                       "pretrain_model_path":"/Volumes/work/project/unlp/unlp/transformers/bert-base-chinese",
    #                                                                       "model_path": "/Volumes/work/project/unlp/unlp/supervised/classification/data/THUCNews/saved_dict/DPCNN.ckpt" })
    # res = service.run_train()
    # predict-DPCNN
    # service = Service(model_type="DPCNN", use_word=True)
    # res = service.run_predict(text=['艺龙网并购两家旅游网站'], model_path='./data/THUCNews/saved_dict/DPCNN.ckpt', embedding="random")

    # train-BERT
    # service = Service(model_name="ERNIE", mode='train', **{"use_word":True, "embedding":"random", "dataset": '/Volumes/work/project/unlp/unlp/supervised/classification/data/THUCNews',
    #                                                       "model_path":"/Volumes/work/project/unlp/unlp/transformers/ernie-1.0",
    #                                                       "resume":False})
    # res = service.run_train()
    # predict-BERT
    service = Service(model_name="BERT", mode='predict', **{"use_word":True, "embedding":"random", "dataset": '/Volumes/work/project/unlp/unlp/supervised/classification/data/THUCNews',
                                                            "model_path":"/Volumes/work/project/unlp/unlp/supervised/classification/data/THUCNews/saved_dict/BERT",
                                                            "resume":True})
    res = service.run_predict(text=['艺龙网并购两家旅游网站',"封基上周溃退 未有明显估值优势"])
    print(res)
