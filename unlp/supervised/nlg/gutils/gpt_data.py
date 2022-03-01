# -*- coding: utf-8 -*-

"""
@Time    : 2022/3/1 3:37 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""

import os
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils import rnn
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from transformers import BertTokenizerFast


class MyDataset(Dataset):
    """
    """

    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_list[index]
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def __len__(self):
        return len(self.input_list)

def load_dataset(vocab_path, data_dir, max_len):
    """
    加载训练集和验证集
    """
    input_list_train = preprocess(vocab_path=vocab_path, data_path=os.path.join(data_dir, 'train.txt'))
    input_list_val = preprocess(vocab_path=vocab_path, data_path=os.path.join(data_dir, 'dev.txt'))
    input_list_test = preprocess(vocab_path=vocab_path, data_path=os.path.join(data_dir, 'test.txt'))

    train_dataset = MyDataset(input_list_train, max_len)
    val_dataset = MyDataset(input_list_val, max_len)
    test_dataset = MyDataset(input_list_test, max_len)

    return train_dataset, val_dataset, test_dataset

def collate_fn(batch):
    input_ids = rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels

def preprocess(vocab_path, data_path):
    """
    对原始语料进行tokenize，将每段对话处理成如下形式："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    """
    # 初始化tokenizer
    tokenizer = BertTokenizerFast(vocab_file=vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id

    # 读取训练数据集
    with open(data_path, 'r') as f:
        data = f.read()

    # 需要区分linux和windows环境下的换行符
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")
    else:
        train_data = data.split("\n\n")

    # 开始进行tokenize
    # 保存所有的对话数据,每条数据的格式为："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    dialogue_len = []  # 记录所有对话tokenize之后的长度，用于统计中位数与均值
    dialogue_list = []
    for index, dialogue in enumerate(tqdm(train_data)):
        if "\r\n" in data:
            utterances = dialogue.split("\r\n")
        else:
            utterances = dialogue.split("\n")

        input_ids = [cls_id]  # 每个dialogue以[CLS]开头
        for utterance in utterances:
            input_ids += tokenizer.encode(utterance, add_special_tokens=False)
            input_ids.append(sep_id)  # 每个utterance之后添加[SEP]，表示utterance结束
        dialogue_len.append(len(input_ids))
        dialogue_list.append(input_ids)
    len_mean = np.mean(dialogue_len)
    len_median = np.median(dialogue_len)
    len_max = np.max(dialogue_len)
    print("mean of dialogue len:{},median of dialogue len:{},max len:{}".format(len_mean, len_median, len_max))
    return dialogue_list

def generate_subset(data_path, subset_size=10000000):
    """
    用于生成训练子集
    :return:
    """
    with open(data_path, "r", encoding="utf8") as f:
        data = f.read()
    dialogues = data.split("\n\n")
    subset_size = min(len(dialogues), subset_size)

    print("generating subset,please wait a few minutes")
    subset_list = []
    for dialogue_index, dialogue in enumerate(dialogues):
        if dialogue_index >= subset_size:
            break
        for utterance in dialogue.split("\n"):
            subset_list.append(utterance)
    return subset_list


def compute_dialogue_length(data_path):
    """
    查看聊天语料中的dialogue的长度分布
    :return:
    """
    with open(data_path, "r", encoding="utf8") as f:
        data = f.read()
    dialogues = data.split("\n\n")
    # 统计各个dialogue的长度
    dialogues_lengths = [len(dialogue.replace("\n", "")) for dialogue in dialogues]
    counter = Counter(dialogues_lengths)  # {label:sum(label)}
    dialogue_length_arr = list(counter)
    num_arr = [counter[element] for element in list(counter)]
    print(counter[300])

    x_major_locator = MultipleLocator(100)  # MultipleLocator用于设置刻度间隔
    # y_major_locator = MultipleLocator(20000)
    ax = plt.gca()  # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为10的倍数
    # ax.yaxis.set_major_locator(y_major_locator)

    plt.xlabel('dialogue length')
    plt.ylabel('number of dialogue')
    # plt.plot(dialogue_length_arr, num_arr, c='green')
    plt.scatter(dialogue_length_arr, num_arr)
    plt.show()

if __name__ == '__main__':
    file_root = os.path.dirname(__file__)
    compute_dialogue_length(os.path.join(file_root, '../data/chnchat/data/train_bak.txt'))