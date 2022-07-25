# -*- coding: utf-8 -*-

"""
@Time    : 2022/3/7 6:16 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
import time
file_root = os.path.dirname(__file__)
import sys
sys.path.append(file_root)
from generate.language_model_generate import BartAugmentor,BertAugmentor
from eda.eda import EDA
from translate.baidu_translate import Translator

class Augmentation(object):
    def __init__(self, mode, model_path='', num_aug=9, **kwargs):
        model_types = ['BERT', "BART", "EDA", "Translate"]
        if mode in model_types:
            self.mode = mode
            self.num_aug=num_aug
            if self.mode == 'BART':
                self.model = BartAugmentor(model_dir=model_path)
            elif self.mode == 'BERT':
                self.model = BertAugmentor(model_dir=model_path)
            elif self.mode == 'EDA':
                self.model = EDA(model_path=model_path)
            else:
                self.model = Translator()
        else:
            print("suport model_types: {}".format("##".join(model_types)))
            os._exit(-1)

    def run(self, text):
        b0 = time.time()
        if self.mode in ['BERT', 'BART', 'EDA']:
            res = self.model.augment(text, num_aug=self.num_aug)
        else:
            res = [self.model.back_translate(text, src_lang='zh', tgt_lang='en')[1]]
        print('cost {}'.format(time.time() - b0))
        return res

if __name__ == '__main__':
    # 预训练时可以通过resume (bool类型）控制是否继续训练，其他predict和evaluate阶段可以不传入这个参数
    DataAugmentation = Augmentation(mode="EDA", model_path="/data/lss/deepenv/deepenv-data/unlp包/transformer/word2vec/light_Tencent_AILab_ChineseEmbedding.bin")
    res = DataAugmentation.run(text="封基上周溃退 未有明显估值优势")
    print(res)