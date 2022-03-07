# -*- coding: utf-8 -*-

"""
@Time    : 2022/2/14 5:52 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""

from unlp.supervised import text_classify
from unlp.supervised import sequence_labeling
from unlp.supervised import text_summarize
from unlp.supervised import dialogue_generation
from unlp.supervised.classification.data import THUCNews as ClassificationDataFormat
from unlp.supervised.ner.data import cluner as NerDataFormat
from unlp.supervised.nlg.data import weibo as SummarizationDataFormat
from unlp.supervised.nlg.data import chnchat as DialogueDataFormat