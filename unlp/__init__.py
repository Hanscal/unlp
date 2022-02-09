#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2021/2/26 12:07 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
import unlp.taskflow
from unlp import unsupervised


name = "unlp"

# 无监督获得关键字
class UkeywordsExtract(object):
    def __init__(self, model_path, model_type):
        if model_type not in ['keybert','tfidf']:
            print("suport model_type: {}".format("##".join(['keybert','tfidf'])))
            os._exit(-1)
        if not model_path and model_type != 'tfidf':
            print('model download automatically!')
        self.model = unsupervised.keywords_extract.Extract(model_path=model_path, model_type=model_type)

    def run(self, text_list:list)->list:
        keywords = self.model.run(text_list)
        # print(keywords)
        return keywords

# 无监督获得相似度
class UTextSimilarity(object):
    def __init__(self, model_path, model_type):
        if model_type not in ['sentbert','w2v']:
            print("suport model_type: {}".format("##".join(['sentbert','w2v'])))
            os._exit(-1)
        if not model_path:
            print('model download automatically!')
        self.model = unsupervised.text_similarity.Similarity(model_path=model_path, model_type=model_type)

    def run(self, text1:str, text2:str):
        sim = self.model.get_score(text1, text2)
        # print(keywords)
        return sim

# 无监督获得嵌入向量
class UTextEmbedding(object):
    def __init__(self, model_path, model_type):
        if model_type not in ['sentbert', 'w2v']:
            print("suport model_type: {}".format("##".join(['sentbert', 'w2v'])))
            os._exit(-1)
        if not model_path:
            print('model download automatically!')
        self.model = unsupervised.text_embedding.Embedding(model_path=model_path, model_type=model_type)

    def run(self, text_list: list):
        emds = self.model.run(text_list)
        return emds

# 无监督获得向量后进行语义搜索
class USemanticSearch(object):
    def __init__(self, model_path, model_type, corpus:list):
        if model_type not in ['sentbert', 'w2v']:
             print("suport model_type: {}".format("##".join(['bm25', 'sentbert', 'w2v'])))
             os._exit(-1)
        if not model_path and model_type != 'bm25':
            print('model download automatically!')
        self.model = unsupervised.semantic_search.SemanticSearch(model_path=model_path, model_type=model_type, corpus=corpus)

    def run(self, query:list, top_k=5):
        res = self.model.run_search(query, top_k)
        return res

