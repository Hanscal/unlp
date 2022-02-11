#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2021/2/26 12:03 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
from unlp import USemanticSearch

def semantic(model_path, model_type, corpus: list, queries:list):
    # model_path为空会自动加载模型
    s = USemanticSearch(model_path=model_path, model_type=model_type, corpus=corpus)
    res = s.run(query=queries, top_k=5)
    print(res)

if __name__ == '__main__':
    # 如果没有模型权重文件，请运行下面注释的函数
    file_root = os.path.dirname(__file__)
    corpus = [
        '花呗更改绑定银行卡',
        '我什么时候开通了花呗',
        'A man is eating food.',
        'A man is eating a piece of bread.',
        'The girl is carrying a baby.',
        'A man is riding a horse.',
        'A woman is playing violin.',
        'Two men pushed carts through the woods.',
        'A man is riding a white horse on an enclosed ground.',
        'A monkey is playing drums.',
        'A cheetah is running behind its prey.'
    ]

    queries = [
        '如何更换花呗绑定银行卡',
        'A man is eating pasta.',
        'Someone in a gorilla costume is playing a set of drums.',
        'A cheetah chases prey on across a field.']

    semantic(model_path=os.path.join(file_root, '../unlp/transformers/paraphrase-multilingual-MiniLM-L12-v2'),
             model_type='sentbert',
             corpus=corpus, queries=queries)
    semantic(model_path=os.path.join(file_root, '../unlp/transformers/word2vec/light_Tencent_AILab_ChineseEmbedding.bin'),
             model_type='w2v',
             corpus=corpus, queries=queries)

    # model_path为空会自动加载模型
    # semantic(model_path='',
    #          model_type='sentbert',
    #          corpus=corpus, queries=queries)
    # semantic(model_path='',
    #          model_type='w2v',
    #          corpus=corpus, queries=queries)