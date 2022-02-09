# -*- coding: utf-8 -*-

"""
@Time    : 2022/1/14 3:34 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
import time
import jieba
from typing import List
file_root = os.path.dirname(__file__)
import sys
sys.path.append(file_root)
from KeyBERT.keybert import KeyBERT
from TFIDF.tfidf import TFIDF

class Extract(object):
    def __init__(self, model_path='',model_type='keybert',**kwargs):
        if model_type =='keybert':
            self.model = KeyBERT(model_path) if os.path.exists(model_path) else KeyBERT()
        elif model_type == 'tfidf':
            self.model = TFIDF(**kwargs)
        else:
            print("suport model_type: {}".format("##".join(['keybert','tfidf'])))
            os._exit(-1)
        self.model_type = model_type

    def run(self, texts:List, **kwargs):
        keywords_list = []
        b0 = time.time()
        if self.model_type =='keybert':
            texts_list = [" ".join(jieba.lcut(t)) for t in texts]
            keywords_list = self.model.extract_keywords(texts_list, **kwargs)
        elif self.model_type == 'tfidf':
            keywords_list = self.model.run_keywords(texts, **kwargs)
        else:
            print("suport model_type: {}".format("##".join(['keybert', 'tfidf'])))
        print('cost {}'.format(time.time() - b0))
        return keywords_list

if __name__ == '__main__':
    a = "第一条  为规范企业国有产权转让行为，推动国有资产存量的合理流动，防止国有资产流失，根据国家有关法律、法规的规定，制定本办法。"
    b = '''该研究主持者之一、波士顿大学地球与环境科学系博士陈池（音）表示，“尽管中国和印度国土面积仅占全球陆地的9%，但两国为这一绿化过程贡献超过三分之一。考虑到人口过多的国家一般存在对土地过度利用的问题，这个发现令人吃惊。”
                                                                   NASA埃姆斯研究中心的科学家拉玛·内曼尼（Rama Nemani）说，“这一长期数据能让我们深入分析地表绿化背后的影响因素。我们一开始以为，植被增加是由于更多二氧化碳排放，导致气候更加温暖、潮湿，适宜生长。”
                                                                   “MODIS的数据让我们能在非常小的尺度上理解这一现象，我们发现人类活动也作出了贡献。'''
    ext = Extract(model_path="/Volumes/work/project/unlp/unlp/transformers/paraphrase-multilingual-MiniLM-L12-v2",
                  model_type='tfidf')
    keywords = ext.run([a,b])
    print(keywords)