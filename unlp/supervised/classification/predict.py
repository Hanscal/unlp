# -*- coding: utf-8 -*-

"""
@Time    : 2022/2/14 5:52 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import json
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

file_root = os.path.dirname(__file__)
sys.path.append(file_root)

from sutils.dutils import TransfomerDatasetIterater, DatasetIterater
from sutils.dutils import char_tokenizer, word_tokenizer

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
CLS = '[CLS]'

class Predict(object):
    def __init__(self,  config, use_word):
        self.build_iterator = self.load_dataset
        if config.model_name in ['BERT', 'ERNIE']:
            self.tokenizer = config.tokenizer
            self.build_iterator = self.load_transformer_dataset
        elif use_word:
            self.tokenizer = word_tokenizer
            self.vocab = json.load(open(config.vocab_path, 'r'))
        else:
            self.tokenizer = char_tokenizer
            self.vocab = json.load(open(config.vocab_path, 'r'))
        self.config = config

    def predict(self, model, texts):
        text_iter = self.build_iterator(texts)
        model.eval()
        predict_all = np.array([], dtype=int)
        with torch.no_grad():
            for i, (texts, _) in enumerate(text_iter):
                outputs = model(texts)
                if outputs.ndim == 1:
                    outputs = torch.unsqueeze(outputs, 0)
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                predict_all = np.append(predict_all, predic)

        return predict_all

    def load_dataset(self, texts, pad_size=32):
        contents = []
        for content in tqdm(texts):
            content = content.strip()
            if not content:
                continue
            words_line = []
            token = self.tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(self.vocab.get(word, self.vocab.get(UNK)))
            contents.append((words_line, 0, seq_len))
        content_iter = DatasetIterater(contents, self.config.batch_size, self.config.device)
        return content_iter


    def load_transformer_dataset(self, texts, pad_size=32):
        contents = []
        for content in tqdm(texts):
            token = self.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = self.tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, 0, seq_len, mask))  # 这里的label值无所谓
        content_iter = TransfomerDatasetIterater(contents, self.config.batch_size, self.config.device)
        return content_iter
