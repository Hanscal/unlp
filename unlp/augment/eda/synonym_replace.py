# -*- coding: utf-8 -*-

"""
@Time    : 2022/3/2 10:32 上午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""

import os
import numpy as np
import jieba
import jieba.analyse
from gensim.models import KeyedVectors
from gensim import matutils


def read_samples(file_path):
    samples = []
    with open(file_path, 'r', encoding='utf8') as file:
        for line in file:
            if not line.strip():
                continue
            samples.append(line.strip())
    return samples

class EmbedReplace(object):
    def __init__(self, wv_path, **kwargs):
        binary = False
        if wv_path.endswith('bin'):
            binary = True
        print("loading word2vec model from {}".format(wv_path))
        self.wv = KeyedVectors.load_word2vec_format(wv_path, binary=binary)

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

    def extract_keywords(self, text, topk=5):
        """ 提取关键词
        :param topk: 前 topk 个关键词
        :return: 返回的关键词列表
        """
        keys = jieba.analyse.extract_tags(text, topK=topk)

        return keys

    def replace_words(self, sample, keywords, ratio=0.2):
        """用wordvector的近义词来替换，并避开关键词
        :param sample (list): reference token list
        :param keywords (list): A reference represented by a word bag model
        :return: 新的文本
        """
        #
        num = int(len(sample) * ratio) # 随机替换30%的单词
        new_tokens = sample.copy()
        indexes = np.random.choice(len(sample), num)
        for index in indexes:
            token = sample[index]
            if self.is_chinese(token) and token not in keywords and token in self.wv:
                new_tokens[index] = self.wv.most_similar(positive=token, negative=None, topn=1)[0][0]

        return new_tokens

    def run_replace(self, sample):
        keys = self.extract_keywords(sample)
        tokens = jieba.lcut(sample)
        res = self.replace_words(tokens, keys)
        return res

if __name__ == '__main__':
    data_dir = '/Volumes/work/project/unlp/unlp/augment/data'
    samples = read_samples(os.path.join(data_dir, 'corpus.txt'))
    wv_path = '/Volumes/work/project/unlp//unlp/transformers/word2vec/light_Tencent_AILab_ChineseEmbedding.bin'
    replacer = EmbedReplace(wv_path)
    res = replacer.run_replace(samples)
    print(res)