#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2021/2/26 12:03 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
from unlp import UTextSimilarity

def similarity(model_path, model_type, texta: str, textb: str):
    # model_path为空会自动加载模型
    s = UTextSimilarity(model_path=model_path, model_type=model_type)
    res = s.run(texta, textb)
    print(res)

if __name__ == '__main__':
    # 如果没有模型权重文件，请运行下面注释的函数
    file_root = os.path.dirname(__file__)
    a = "第一条  为规范企业国有产权转让行为，推动国有资产存量的合理流动，防止国有资产流失，根据国家有关法律、法规的规定，制定本办法。"
    b = '''该研究主持者之一、波士顿大学地球与环境科学系博士陈池（音）表示，“尽管中国和印度国土面积仅占全球陆地的9%，但两国为这一绿化过程贡献超过三分之一。考虑到人口过多的国家一般存在对土地过度利用的问题，这个发现令人吃惊。
              NASA埃姆斯研究中心的科学家拉玛·内曼尼（Rama Nemani）说，“这一长期数据能让我们深入分析地表绿化背后的影响因素。我们一开始以为，植被增加是由于更多二氧化碳排放，导致气候更加温暖、潮湿，适宜生长。
              MODIS的数据让我们能在非常小的尺度上理解这一现象，我们发现人类活动也作出了贡献。'''
    similarity(model_path=os.path.join(file_root, '../unlp/transformers/paraphrase-multilingual-MiniLM-L12-v2'),
               model_type='sentbert',
               texta=a, textb=b)
    similarity(model_path=os.path.join(file_root, '../unlp/transformers/word2vec/light_Tencent_AILab_ChineseEmbedding.bin'),
               model_type='w2v', texta='花呗',textb='支付宝')

    # model_path为空会自动加载模型
    # similarity(model_path='',
    #            model_type='sentbert',
    #            texta=a, textb=b)
    # similarity(model_path='',
    #            model_type='w2v', texta='花呗',textb='支付宝')