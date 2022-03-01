# -*- coding: utf-8 -*-
import os
import sys
import time

import setproctitle
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)

from gmodels.model import Model
from gmodels.adagrad_custom import AdagradCustom
from gutils.batcher import Batcher
from gutils.batcher import get_input_from_batch, get_output_from_batch

from gutils.data import Vocab
from gutils.utils import calc_running_avg_loss
from gutils.config import get_argparse

from eval import Evaluate

setproctitle.setproctitle('summary_generate')

def init_print(config):
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    print("时间:{}".format(stamp))
    print("***参数:***")
    for k, v in config.__dict__.items():
        if not k.startswith("__"):
            print(":".join([k, str(v)]))

class Train(object):
    def __init__(self, **kwargs):
        config = get_argparse()
        args_bak = vars(config)
        for k, v in kwargs.items():
            if k in args_bak:
                args_bak[k] = v
        # set cuda and device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args_bak['device'] = device

        # 数据目录
        data_dir = os.path.join(config.data_dir, 'data') if not config.data_dir.endswith('data') else config.data_dir
        self.vocab = Vocab(os.path.join(data_dir, 'vocab.txt'), config.vocab_size)
        self.batcher = Batcher(os.path.join(data_dir, 'train.json'), self.vocab, mode='train', config=config, single_pass=False)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        # log dir
        log_dir = os.path.join(config.data_dir, 'log', config.model_type)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_file_path = os.path.join(log_dir, 'train_{}'.format(stamp))

        # model_save_dir
        model_save_dir = os.path.join(config.data_dir, 'saved_dict', config.model_type)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        self.config = config
        init_print(self.config)

        self.iter_step, self.running_avg_loss = self.setup_train()
        self.model_type = config.model_type
        self.model_dir = model_save_dir

    def save_model(self, running_avg_loss, iter_step):
        """保存模型"""
        state = {
            'iter': iter_step,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, '{}.pt'.format(self.model_type))
        torch.save(state, model_save_path)


    def setup_train(self):
        """模型初始化或加载、初始化迭代次数、损失、优化器"""
        # 初始化模型
        self.model = Model(self.config, is_eval=False)
        # 模型参数的列表
        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        # 定义优化器
        # self.optimizer = optim.Adam(params, lr=config.adam_lr)
        # 使用AdagradCustom做优化器
        initial_lr = self.config.lr_coverage if self.config.is_coverage else self.config.lr
        self.optimizer = AdagradCustom(params, lr=initial_lr, initial_accumulator_value=self.config.adagrad_init_acc)
        # 初始化迭代次数和损失
        start_iter, start_loss = 0, 0
        # 如果传入的已存在的模型路径，加载模型继续训练
        if os.path.exists(self.config.model_path):
            state = torch.load(self.config.model_path, map_location = lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not self.config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(self.config.device)

        return start_iter, start_loss

    def train_one_batch(self, batch):
        """
        训练一个batch，返回该batch的loss。
        enc_batch:             torch.Size([16, 400]), 16篇文章的编码，不足400词的用pad的编码补足, oov词汇用0编码；
        enc_padding_mask:      torch.Size([16, 400]), 对应pad的位置为0，其余为1；
        enc_lens:              numpy.ndarray, 列表内每个元素表示每篇article的单词数；
        enc_batch_extend_vocab:torch.Size([16, 400]), 16篇文章的编码;oov词汇用超过词汇表的编码；
        extra_zeros:           torch.Size([16, 文章oov词汇数量]) zero tensor;
        c_t_1:                 torch.Size([16, 512]) zero tensor;
        coverage:              Variable(torch.zeros(batch_size, max_enc_seq_len)) if is_coverage==True else None;coverage模式时后续有值
        ----------------------------------------
        dec_batch:             torch.Size([16, 100]) 摘要编码含有开始符号编码以及PAD；
        dec_padding_mask:      torch.Size([16, 100]) 对应pad的位置为0，其余为1；
        max_dec_len:           标量，摘要词语数量，不包含pad
        dec_lens_var:          torch.Size([16] 摘要词汇数量         
        target_batch:          torch.Size([16, 100]) 目标摘要编码含有STOP符号编码以及PAD
        """
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(self.config, batch)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(self.config, batch)
        self.optimizer.zero_grad()

        # [B, max(seq_lens), 2*hid_dim], [B*max(seq_lens), 2*hid_dim], tuple([2, B, hid_dim], [2, B, hid_dim])
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)   # (h,c) = ([1, B, hid_dim], [1, B, hid_dim])
        step_losses = []
        for di in range(min(max_dec_len, self.config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]      # 摘要的一个单词，batch里的每个句子的同一位置的单词编码
            # print("y_t_1:", y_t_1, y_t_1.size())
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage, di)
            target = target_batch[:, di]  # 摘要的下一个单词的编码
            # print("target-iter:", target, target.size())
            # print("final_dist:", final_dist, final_dist.size())
            # input("go on>>")
            # final_dist 是词汇表每个单词的概率，词汇表是扩展之后的词汇表，也就是大于预设的50_000
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()   # 取出目标单词的概率gold_probs
            step_loss = -torch.log(gold_probs + self.config.eps)  # 最大化gold_probs，也就是最小化step_loss（添加负号）
            if self.config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + self.config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
                
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), self.config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), self.config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), self.config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def train(self):
        # 训练设置，包括
        start = time.time()
        loss_eval = 10

        for _ in tqdm(range(self.config.num_train_epochs), desc="Training epoch"):
            for _ in tqdm(range(self.config.max_iter_steps_epoch), desc="Steps per epoch"):
                # 获取下一个batch数据
                batch = self.batcher.next_batch()
                loss = self.train_one_batch(batch)

                self.running_avg_loss = calc_running_avg_loss(loss, self.running_avg_loss)
                self.iter_step += 1

                # print_interval = 100
                if self.iter_step % self.config.logging_steps == 0:
                    # lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    print('steps %d, seconds for %d steps: %.2f, loss: %f' % (self.iter_step, self.config.logging_steps, time.time() - start, loss))
                    start = time.time()

                # 100次迭代就保存一下模型
                if self.iter_step % self.config.save_steps == 0:
                    evaluate = Evaluate(model=self.model,**{"data_dir":self.config.data_dir})
                    loss_dev = evaluate.run_eval()
                    if loss_dev < loss_eval:
                        loss_eval = loss_dev
                    self.save_model(self.running_avg_loss, self.iter_step)
        return self.running_avg_loss

    def rouge_cal(self, target_batch, predict):
        # 下面是求解rouge相关
        # target_batch, decode_batch
        for i, dec in enumerate(target_batch):
            tgt = target_batch[i]
            tgt_str = ''.join([self.vocab._id_to_word.get(int(w), '[UNK]') for w in tgt if
                               int(w) not in [self.vocab._word_to_id[_] for _ in ['[PAD]', '[START]', '[STOP]']]])
            article = predict.build_batch_by_article(tgt_str, self.vocab, self.config.beam_size)
            pred = self.predict.run_predict([article])
            rouge_2 = self.avg_rouge(tgt, pred, 2)

    def ngram(self, text, n):
        leng = len(text)
        word_dic = {}
        for i in range(0, leng, n):
            start = i
            words = ""
            if leng - start < n:
                break
            else:
                words = text[start: start + n]
                word_dic[words] = 1
        return word_dic

    def compute_rouge_n(self, text1, text2, n):
        dic1 = self.ngram(text1, n)
        dic2 = self.ngram(text2, n)
        x = 0
        y = len(dic2)
        for w in dic1:
            if w in dic2:
                x += 1
        rouge = x / y
        return rouge if rouge <= 1.0 else 1.0

    def avg_rouge(self, ref_batch, dec_batch, n):
        scores_list = []
        for ref_cont, dec_cont in zip(ref_batch, dec_batch): # ->str, contain space
            score = self.compute_rouge_n(dec_cont, ref_cont, n)
            scores_list.append(score)
        return sum(scores_list) / len(scores_list)


if __name__ == '__main__':
    kwargs = {"data_dir":os.path.join(os.path.dirname(__file__),'data/weibo')}
    train_processor = Train(**kwargs)
    loss = train_processor.train()
