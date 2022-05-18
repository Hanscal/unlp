# -*- coding: utf-8 -*-

"""
@Time    : 2022/3/1 9:35 上午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
# 回译

import os
import numpy as np
import jieba
import jieba.analyse
from gensim.models import KeyedVectors
from gensim import matutils


def read_samples(file_path):
    """
    :param file_path: 文本文件的路径
     读入文本数据
    """
    samples = []
    with open(file_path, 'r', encoding='utf8') as file:
        for line in file:
            if not line.strip():
                continue
            samples.append(line.strip())
    return samples


class EmbedReplace(object):
    """
     EmbedReplace: 同义词替换
    """
    def __init__(self, wv_path, **kwargs):
        """
        :param wv_path: w2v模型路径
         加载w2v模型 KeyedVectors可以完成相似性查找
        """
        binary = False
        if wv_path.endswith('bin'):
            binary = True
        print("loading word2vec model from {}".format(wv_path))
        # load_word2vec_format: 加载 save_word2vector_format 后的模型
        # {word: vector}
        self.wv = KeyedVectors.load_word2vec_format(wv_path, binary=binary)

    def is_chinese(self, word):
        """
        :param word: 词组
        :return: True / False
         判断是否含有中文字符
        """
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False

    def vectorize(self, docs, vocab_size):
        """
        :param docs: 语料库
        :param vocab_size: 词表大小
        """
        # Convert corpus into a dense numpy 2D array, with documents as columns.
        return matutils.corpus2dense(docs, vocab_size)

    def extract_keywords(self, text, topk=5):
        """
        :param topk: 前 topk 个关键词
        :return: topk个关键词组成的列表
         提取关键词
        """
        # 使用TF-IDF算法提取关键词
        keys = jieba.analyse.extract_tags(text, topK=topk)
        return keys

    def replace_words(self, sample, keywords, ratio=0.2):
        """
        :param sample: 参考token列表
        :param keywords: 关键词表（防止关键词被替换影响句义）
        :param ratio: 替换比率
        :return: 新的文本
         用word2vector的近义词来替换，并避开关键词
        """
        num = int(len(sample) * ratio)  # 随机替换20%的单词
        new_tokens = sample.copy()

        indexes = np.random.choice(len(sample), num)
        for index in indexes:
            token = sample[index]
            # sample[index]是中文字符 且 sample[index]不在关键词列表中 且 sample[index]在w2v模型生成的词表中
            if self.is_chinese(token) and token not in keywords and token in self.wv:
                new_tokens[index] = self.wv.most_similar(positive=token, negative=None, topn=1)[0][0]

        return new_tokens

    def run_replace(self, sample):
        """
        :param sample: read_samples('corpus.txt')
        :return: replace(cut(extract(sample)))
         同义词替换过程
        """
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
