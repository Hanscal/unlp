#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2021/2/26 12:03 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import numpy as np

def get_similarity(query_vec, vec_list, sent, sent_list, metric_type='cos'):
    similarity = None
    if metric_type == 'cos':
        vec_arr = np.asarray(vec_list)
        query_arr = np.asarray(query_vec)
        similarity_arr = np.dot(vec_arr, query_arr.reshape(1, -1).T)

        # 当target 文本与文本库中的某些语句完全一样的话，将其similarity score改为1
        indices = [i for i, s in enumerate(sent_list) if sent == s]
        for i in indices:
            similarity_arr[i] = [1]

        similarity_arr_arg = np.argsort(similarity_arr, axis=0)[::-1]  # 从大到小排序
        similarity = [(similarity_arr[i][0][0], i[0]) for i in similarity_arr_arg]
    else:
        print('not support metric type in similarity get!')
    return similarity


if __name__ == '__main__':
    # filepath = '/Volumes/work/project/article_search/data/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.bin'
    # model = textembedding.load_word2vect(filepath)
    # word_vect = textembedding.get_word_embedding(model, '中国')
    # sent_vect = textembedding.get_sentence_embedding(model, '我是中国人，我爱我的祖国。', stop_words_path='')
    # vect_sim = textembedding.get_vector_similarity(word_vect,[sent_vect, word_vect])
    # print(model.vector_size)
    # print(vect_sim)
    # vect_sim = get_similarity(word_vect,[word_vect,sent_vect],'中国',['中国','我是中国人，我爱我的祖国。'])
    # print(vect_sim)

    from unlp import UkeywordsExtract, USemanticSearch, UTextEmbedding, UTextSimilarity

    s = UTextSimilarity(
        model_path='/unlp/transformers/paraphrase-multilingual-MiniLM-L12-v2',
        model_type='sentbert')
    a = "第一条  为规范企业国有产权转让行为，推动国有资产存量的合理流动，防止国有资产流失，根据国家有关法律、法规的规定，制定本办法。"
    b = '''该研究主持者之一、波士顿大学地球与环境科学系博士陈池（音）表示，“尽管中国和印度国土面积仅占全球陆地的9%，但两国为这一绿化过程贡献超过三分之一。考虑到人口过多的国家一般存在对土地过度利用的问题，这个发现令人吃惊。”
                                                                           NASA埃姆斯研究中心的科学家拉玛·内曼尼（Rama Nemani）说，“这一长期数据能让我们深入分析地表绿化背后的影响因素。我们一开始以为，植被增加是由于更多二氧化碳排放，导致气候更加温暖、潮湿，适宜生长。”
                                                                           “MODIS的数据让我们能在非常小的尺度上理解这一现象，我们发现人类活动也作出了贡献。'''
    res = s.run(a, b)
    print(res)