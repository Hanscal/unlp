# -*- coding: utf-8 -*-

"""
@Time    : 2022/3/2 2:46 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
import sys
import jieba
import synonyms
import random
from random import shuffle

random.seed(2022)

file_root = os.path.dirname(__file__)
sys.path.append(file_root)
from synonym_replace import EmbedReplace

class EDA(object):
    def __init__(self, **kwargs):
        # 停用词列表，默认使用哈工大停用词表
        f = open(os.path.join(file_root, 'stopwords/hit_stopwords.txt'))
        self.stop_words = list()
        for stop_word in f.readlines():
            self.stop_words.append(stop_word.strip())
        wv_path = kwargs.get('wv_path', '')
        self.synonym_replacer = None
        if os.path.exists(wv_path):
            self.synonym_replacer = EmbedReplace(wv_path)

    ########################################################################
    # 同义词替换
    # 替换一个语句中的n个单词为其同义词
    ########################################################################
    def synonym_replacement(self, words, n):
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break

        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')

        return new_words


    def get_synonyms(self, word):
        return synonyms.nearby(word)[0]


    ########################################################################
    # 随机插入
    # 随机在语句中插入n个词
    ########################################################################
    def random_insertion(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words


    def add_word(self, new_words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[random.randint(0, len(new_words) - 1)]
            synonyms = self.get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = random.choice(synonyms)
        random_idx = random.randint(0, len(new_words) - 1)
        new_words.insert(random_idx, random_synonym)


    ########################################################################
    # Random swap
    # Randomly swap two words in the sentence n times
    ########################################################################

    def random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words


    def swap_word(self, new_words):
        random_idx_1 = random.randint(0, len(new_words) - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words) - 1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
        return new_words


    ########################################################################
    # 随机删除
    # 以概率p删除语句中的词
    ########################################################################
    def random_deletion(self, words, p):
        if len(words) == 1:
            return words

        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)

        if len(new_words) == 0:
            rand_int = random.randint(0, len(words) - 1)
            return [words[rand_int]]

        return new_words


    ########################################################################
    # EDA函数
    def run(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
        seg_list = jieba.cut(sentence)
        seg_list = " ".join(seg_list)
        words = list(seg_list.split())
        num_words = len(words)

        augmented_sentences = []
        num_new_per_technique = int(num_aug / 4) + 1
        n_sr = max(1, int(alpha_sr * num_words))
        n_ri = max(1, int(alpha_ri * num_words))
        n_rs = max(1, int(alpha_rs * num_words))

        # print(words, "\n")

        # 同义词替换sr
        for _ in range(num_new_per_technique):
            a_words = self.synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words))

        # 词向量同义词替换er
        if self.synonym_replacer is not None:
            for _ in range(num_new_per_technique*2): # 如果用了这个方法，则产生两倍的数量
                a_words = self.synonym_replacer.run_replace(sentence)
                augmented_sentences.append(' '.join(a_words))

        # 随机插入ri
        for _ in range(num_new_per_technique):
            a_words = self.random_insertion(words, n_ri)
            augmented_sentences.append(' '.join(a_words))

        # 随机交换rs
        for _ in range(num_new_per_technique):
            a_words = self.random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))

        # 随机删除rd
        for _ in range(num_new_per_technique):
            a_words = self.random_deletion(words, p_rd)
            augmented_sentences.append(' '.join(a_words))

        # print(augmented_sentences)
        shuffle(augmented_sentences)

        if num_aug >= 1:
            augmented_sentences = augmented_sentences[:num_aug]
        else:
            keep_prob = num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

        augmented_sentences.append(seg_list)

        return augmented_sentences

if __name__ == '__main__':
    kwargs = {"wv_path":'/Volumes/work/project/unlp//unlp/transformers/word2vec/light_Tencent_AILab_ChineseEmbedding.bin'}
    eda = EDA(**kwargs)
    res = eda.run(sentence="我们就像蒲公英，我也祈祷着能和你飞去同一片土地", num_aug=4)
    print(len(res), res)