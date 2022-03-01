# -*- coding: utf-8 -*-

import os
import time
import sys
import torch

sys.path.append(os.path.dirname(__file__))
from gutils.config import get_argparse
from gutils.data import Vocab

from gutils.utils import calc_running_avg_loss
from gutils.batcher import Batcher, Batch
from gutils.batcher import get_input_from_batch
from gutils.batcher import get_output_from_batch

from gmodels.model import Model

class Evaluate(object):
    def __init__(self, model=None, **kwargs):
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
        self.batcher = Batcher(os.path.join(data_dir, 'dev.json'), self.vocab, mode='eval', config=config, single_pass=True)
        if model is None:
            self.model = Model(config, is_eval=True)
        else:
            self.model = model
        self.config = config

    def eval_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(self.config, batch)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(self.config, batch)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, self.config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + self.config.eps)
            if self.config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + self.config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_step_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        return loss.item()

    def run_eval(self):
        running_avg_loss, iter_step = 0, 0
        start = time.time()
        batch = self.batcher.next_batch()
        while batch:
            loss = self.eval_one_batch(batch)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss)
            iter_step += 1

            if iter_step % self.config.logging_steps == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (
                iter_step, self.config.logging_steps, time.time() - start, running_avg_loss))
                start = time.time()
            batch = self.batcher.next_batch()
        return running_avg_loss

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

    def run_rouge(self, refs, decs, n):
        scores_list = []
        for ref_cont, dec_cont in zip(refs, decs): # ->str, contain space
            score = self.compute_rouge_n(dec_cont, ref_cont, n)
            scores_list.append(score)
        return sum(scores_list) / len(scores_list)

if __name__ == '__main__':
    kwargs = {'model_path':"/Volumes/work/project/unlp/unlp/supervised/nlg/data/weibo/saved_dict/point-net/point-net.pt",
              "data_dir":os.path.join(os.path.dirname(__file__),'data/weibo')}
    eval_processor = Evaluate(**kwargs)
    res = eval_processor.run_eval()
    print('loss',res)

    res = eval_processor.run_rouge(["近日，一段消防员用叉子吃饭的视频在网上引起热议。"], ["近日，一段消防员用叉子吃饭的视频在网上引起热议。"], 2)
    print("rouge", res)
