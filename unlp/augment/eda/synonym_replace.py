# -*- coding: utf-8 -*-

"""
@Time    : 2022/3/2 10:32 上午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""

import os
import jieba
import jieba.analyse
import numpy as np

from gensim.models import KeyedVectors, TfidfModel
from gensim.corpora import Dictionary
from gensim import matutils
from itertools import islice

class EmbedReplace(object):
    def __init__(self, wv_path, data_dir, **kwargs):
        binary = False
        if wv_path.endswith('bin'):
            binary = True
        print("loading word2vec model from {}".format(wv_path))
        self.wv = KeyedVectors.load_word2vec_format(wv_path, binary=binary)
        sample_path = os.path.join(data_dir, 'corpus.txt')
        print('reading samples from {}'.format(sample_path))
        self.samples = self.read_samples(sample_path)
        self.samples = [list(jieba.cut(sample)) for sample in self.samples]
        tfidf_path = os.path.join(data_dir, 'tfidf.model')
        dct_path = os.path.join(data_dir, 'tfidf.dict')
        if os.path.exists(tfidf_path):
            self.tfidf_model = TfidfModel.load(tfidf_path)
            self.dct = Dictionary.load(dct_path)
            self.corpus = [self.dct.doc2bow(doc) for doc in self.samples]
        else:
            self.dct = Dictionary(self.samples)
            self.corpus = [self.dct.doc2bow(doc) for doc in self.samples]
            self.tfidf_model = TfidfModel(self.corpus)
            self.dct.save(dct_path)
            self.tfidf_model.save(tfidf_path)
            self.vocab_size = len(self.dct.token2id)

    def read_samples(self, file_path):
        samples = []
        with open(file_path, 'r', encoding='utf8') as file:
            for line in file:
                if not line.strip():
                    continue
                samples.append(line.strip())
        return samples

    def is_chinese(self, word):
        """是否为中文字符
        :param word:
        :return:
        """
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False

    def vectorize(self, docs, vocab_size):
        return matutils.corpus2dense(docs, vocab_size)

    def extract_keywords(self, dct, tfidf, threshold=0.2, topk=5):
        """ 提取关键词
        :param dct (Dictionary): gensim.corpora.Dictionary
        :param tfidf (list):
        :param threshold: tfidf的临界值
        :param topk: 前 topk 个关键词
        :return: 返回的关键词列表
        """
        tfidf = sorted(tfidf, key=lambda x: x[1], reverse=True)

        return list(islice([dct[w] for w, score in tfidf if score > threshold], topk))

    def replace_words(self, sample, doc):
        """用wordvector的近义词来替换，并避开关键词
        :param sample (list): reference token list
        :param doc (list): A reference represented by a word bag model
        :return: 新的文本
        """
        keywords = self.extract_keywords(self.dct, self.tfidf_model[doc])
        #
        num = int(len(sample) * 0.3)
        new_tokens = sample.copy()
        indexes = np.random.choice(len(sample), num)
        for index in indexes:
            token = sample[index]
            if self.is_chinese(token) and token not in keywords and token in self.wv:
                new_tokens[index] = self.wv.most_similar(positive=token, negative=None, topn=1)[0][0]

        return ''.join(new_tokens)

    def run_replace(self):
        res = []
        for sample, doc in zip(self.samples, self.corpus):
            res.append(self.replace_words(sample, doc))
        return res

if __name__ == '__main__':
    data_dir = '/unlp/augment/data'
    wv_path = '/unlp/transformers/word2vec/light_Tencent_AILab_ChineseEmbedding.bin'
    replacer = EmbedReplace(wv_path, data_dir)
    res = replacer.run_replace()
    print(res)