# -*- coding: utf-8 -*-

"""
@Time    : 2022/3/1 4:37 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
import sys
import torch
import logging
from datetime import datetime
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from gutils.config import get_gpt_args
from gmodels.gpt_model import create_model
from gutils.gpt_data import collate_fn, load_dataset


class Evaluator(object):
    def __init__(self, **kwargs):
        config = get_gpt_args(**kwargs)

        # set cuda and device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.device = device

        # 数据目录
        data_dir = os.path.join(config.data_dir, 'data') if not config.data_dir.endswith('data') else config.data_dir
        self.vocab_path = os.path.join(config.model_path, 'vocab.txt')
        self.validate_dataset = load_dataset(self.vocab_path, data_dir, config.max_len, mode='eval')
        self.validate_dataloader = DataLoader(self.validate_dataset, batch_size=self.config.batch_size, shuffle=True,
                                              num_workers=self.config.num_workers, collate_fn=collate_fn, drop_last=False)
        self.config = config

    def validate_epoch(self):
        logging.info("start validating")
        self.model.eval()

        device = self.config.device
        epoch_start_time = datetime.now()
        total_loss = 0
        # 捕获cuda out of memory exception
        try:
            with torch.no_grad():
                for batch_idx, (input_ids, labels) in enumerate(self.validate_dataloader):
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    outputs = self.model.forward(input_ids, labels=labels)
                    logits = outputs.logits
                    loss = outputs.loss
                    loss = loss.mean()

                    total_loss += loss.item()
                    del input_ids, outputs

                # 记录当前epoch的平均loss
                epoch_mean_loss = total_loss / len(self.validate_dataloader)
                logging.info("validate : loss {}".format(epoch_mean_loss))
                epoch_finish_time = datetime.now()
                logging.info('time for validating one epoch: {}'.format(epoch_finish_time - epoch_start_time))
                return epoch_mean_loss

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logging.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logging.info(str(exception))
                raise exception

    def run_eval(self, model=None):
        if model is None:
            self.model = create_model(self.config)
        else:
            self.model = model
        loss = self.validate_epoch()
        return loss

if __name__ == '__main__':
    pass
