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
from torch.utils.data import DataLoader, SequentialSampler

file_root = os.path.dirname(__file__)
sys.path.append(file_root)

from ner_train import NERTrainer
from ner_evaluate import NEREvaluator
from ner_predict import NERPredictor

from nmodels.transformer import WEIGHTS_NAME, BertConfig, AlbertConfig
from processors.utils_ner import CNerTokenizer
from nmodels.bert_for_ner import BertCrfForNer
from nmodels.albert_for_ner import AlbertCrfForNer

from processors.ner_seq import collate_fn
from processors.ner_seq import ner_processors as processors

from tools.dutils import load_and_cache_examples
from tools.config import get_argparse

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert-base-chinese': (BertConfig, BertCrfForNer, CNerTokenizer),
    'chinese-bert-wwm-ext': (BertConfig, BertCrfForNer, CNerTokenizer),
    'ernie-1.0': (BertConfig, BertCrfForNer, CNerTokenizer),
    'albert-base-chinese': (AlbertConfig, AlbertCrfForNer, CNerTokenizer),
}

class Service(object):
    def __init__(self, model_name, mode, **kwargs):
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样

        model_type = model_name

        if mode == 'train':
            model_name_or_path = kwargs['model_path']
            data_dir = kwargs['data_dir']
            self.trainer = NERTrainer(**{"model_name_or_path":model_name_or_path, "resume":kwargs.get('resume',""),
                                    "data_dir":data_dir, "model_type":model_type})

        elif mode == 'evaluate':
            # load eval dataset
            model_name_or_path = kwargs['model_path']
            data_dir = kwargs['data_dir']
            print("loading model from {}".format(model_name_or_path))
            config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
            processor = processors['cluener']()
            data_dir = os.path.join(data_dir, 'data') if not data_dir.endswith('data') else data_dir

            label_list = processor.get_labels(data_dir=data_dir)
            config = config_class.from_pretrained(model_name_or_path, num_labels=len(label_list))
            self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
            self.model = model_class.from_pretrained(model_name_or_path,
                                                     from_tf=bool(".ckpt" in model_name_or_path),
                                                     config=config)
            self.evaluator = NEREvaluator(**{"data_dir":data_dir})

        elif mode == 'predict':
            model_name_or_path = kwargs['model_path']
            data_dir = kwargs['data_dir']
            self.predictor = NERPredictor(**{"model_name_or_path":model_name_or_path,
                                    "data_dir":data_dir, "model_type":model_type})
        self.kwargs = kwargs

    def run_train(self):
        res = self.trainer.train()
        return res

    def run_evaluate(self):
        args = get_argparse().parse_args()
        # 利用kwargs来更新args
        args_bak = vars(args)
        for k, v in self.kwargs.items():
            if k in args_bak:
                args_bak[k] = v

        eval_dataset = load_and_cache_examples(args, 'cluener', self.tokenizer, data_type='dev')
        args.n_gpu = torch.cuda.device_count()
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn)

        res = self.evaluator.evaluate(self.model, eval_dataloader)
        return res

    def run_predict(self, text: List[str]):
        res = self.predictor.predict(texts=text)
        return res


if __name__ == '__main__':
    # train-BERT
    # service = Service(model_name="bert-base-chinese", mode='train', **{"data_dir": '/Volumes/work/project/unlp/unlp/supervised/ner/data/cluner',
    #                                                       "model_path":"/Volumes/work/project/unlp/unlp/transformers/ernie-1.0",
    #                                                       "resume":False})
    # res = service.run_train()

    # evaluate-BERT
    # service = Service(model_name="bert-base-chinese", mode='evaluate', **{"data_dir": '/Volumes/work/project/unlp/unlp/supervised/ner/data/cluner',
    #                                                         "model_path":"/Volumes/work/project/unlp/unlp/supervised/ner/data/cluner/saved_dict/bert-base-chinese"})
    # res = service.run_evaluate()

    # predict-BERT
    service = Service(model_name="bert-base-chinese", mode='predict', **{"data_dir": '/Volumes/work/project/unlp/unlp/supervised/ner/data/cluner',
                                                            "model_path":"/Volumes/work/project/unlp/unlp/supervised/ner/data/cluner/saved_dict/bert-base-chinese"})
    res = service.run_predict(text=['在豪门被多线作战拖累时，正是他们悄悄追赶上来的大好时机。重新找回全队的凝聚力是拉科赢球的资本。',"不久后，“星展中国”南宁分行也将择机开业。除新加坡星展银行外，多家外资银行最近纷纷扩大在华投资，"])
    print(res)