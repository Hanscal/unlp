# -*- coding: utf-8 -*-

"""
@Time    : 2022/3/1 3:53 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
import sys

import torch
from torch.nn import functional as F
from transformers import BertTokenizerFast, GPT2LMHeadModel

sys.path.append(os.path.dirname(__file__))
from gutils.config import set_gpt_args

CONFIG_NAME = 'config.json'

class Predictor(object):
    def __init__(self, **kwargs):
        # 初始化参数
        config = set_gpt_args(**kwargs)
        # set cuda and device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.device = device

        # 初始化tokenizer
        self.vocab_path = os.path.join(config.model_path, 'vocab.txt')
        self.tokenizer = BertTokenizerFast(vocab_file=self.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")

        self.model = GPT2LMHeadModel.from_pretrained(config.model_path)
        self.model = self.model.to(device)
        self.model.eval()
        assert self.model.config.vocab_size == self.tokenizer.vocab_size

        self.config = config

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocab size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
            # ...表示其他维度由计算机自行推断
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits


    def predict(self):
        # 存储聊天记录，每个utterance以token的id的形式进行存储
        if self.config.save_samples_path:
            if not os.path.exists(self.config.save_samples_path):
                os.makedirs(self.config.save_samples_path)
            samples_file = open(self.config.save_samples_path + '/samples.txt', 'a', encoding='utf8')

        history = []
        print('开始和chatbot聊天，输入CTRL + Z以退出')

        while True:
            try:
                text = input("user:")
                # text = "你好"
                if self.config.save_samples_path:
                    samples_file.write("user:{}\n".format(text))

                text_ids = self.tokenizer.encode(text, add_special_tokens=False)
                history.append(text_ids)
                input_ids = [self.tokenizer.cls_token_id]  # 每个input以[CLS]为开头

                for history_id, history_utr in enumerate(history[-self.config.max_history_len:]):
                    input_ids.extend(history_utr)
                    input_ids.append(self.config.tokenizer.sep_token_id)
                input_ids = torch.tensor(input_ids).long().to(self.config.device)
                input_ids = input_ids.unsqueeze(0)
                response = []  # 根据context，生成的response
                # 最多生成max_len个token
                for _ in range(self.config.max_len):
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs.logits
                    next_token_logits = logits[0, -1, :]
                    # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                    for id in set(response):
                        next_token_logits[id] /= self.config.repetition_penalty
                    next_token_logits = next_token_logits / self.config.temperature
                    # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                    next_token_logits[self.tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                    filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=self.config.topk, top_p=self.config.topp)
                    # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    if next_token == self.tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                        break
                    response.append(next_token.item())
                    input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
                    # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
                    # print("his_text:{}".format(his_text))
                history.append(response)
                text = self.tokenizer.convert_ids_to_tokens(response)
                print("chatbot:" + "".join(text))
                if self.config.save_samples_path:
                    samples_file.write("chatbot:{}\n".format("".join(text)))
            except KeyboardInterrupt:
                if self.config.save_samples_path:
                    samples_file.close()
                break



if __name__ == '__main__':
    kwargs = {"data_dir":os.path.join(os.path.dirname(__file__),'data/chnchat'),
              "model_path":"/Volumes/work/project/unlp/unlp/transformers/gpt2-chnchit"}
    t = Predictor(**kwargs)
    t.predict()