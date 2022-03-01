# -*- coding: utf-8 -*-

"""
@Time    : 2022/3/1 3:53 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
import shutil
import sys
import time

import torch
from torch.nn import DataParallel
import transformers
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from datetime import datetime
from gmodels.gpt_model import calculate_acc, EarlyStopping, create_model
from gutils.gpt_data import collate_fn, load_dataset
from gutils.utils import create_logger
from gutils.config import set_gpt_args
from gpt_eval import validate_epoch

CONFIG_NAME = 'config.json'

class Trainer(object):
    def __init__(self, **kwargs):
        # 初始化参数
        config = set_gpt_args(**kwargs)
        # set cuda and device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.device = device

        # 数据目录
        data_dir = os.path.join(config.data_dir, 'data') if not config.data_dir.endswith('data') else config.data_dir

        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # log dir
        log_dir = os.path.join(config.data_dir, 'log', config.model_type)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file_path = os.path.join(log_dir, 'train_{}'.format(stamp))
        self.logger = create_logger(log_file_path)

        # model_save_dir
        self.model_save_dir = os.path.join(config.data_dir, 'saved_dict', config.model_type)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)


        # 创建模型
        # 如果需要从头训练，则需要提供config.json文件
        if not os.path.exists(config.model_path):
            assert os.path.exists(config.model_config), "need to provide model path or provide model_config training from scratch!"

        # 初始化tokenizer
        self.vocab_path = os.path.join(config.model_path, 'vocab.txt')
        self.tokenizer = BertTokenizerFast(vocab_file=self.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
        config.sep_id = self.tokenizer.sep_token_id
        config.pad_id = self.tokenizer.pad_token_id
        config.cls_id = self.tokenizer.cls_token_id
        vocab_size = self.tokenizer.vocab_size

        self.model = create_model(config, vocab_size)
        assert self.model.config.vocab_size == self.tokenizer.vocab_size

        # 加载训练集和验证集
        # ========= Loading Dataset ========= #
        self.train_dataset, self.validate_dataset, self.test_dataset = load_dataset(self.vocab_path, data_dir, config.max_len)

        # save config.json to model_save_dir
        shutil.copyfile(os.path.join(config.model_path, CONFIG_NAME), os.path.join(self.model_save_dir, CONFIG_NAME))

        # 并行训练模型
        if torch.cuda.device_count() > 1:
            self.model = DataParallel(self.model).cuda()
            # model = BalancedDataParallel(args.gpu0_bsz, model, dim=0).cuda()
            self.logger.info("use GPU {} to train".format(config.device))


        # 计算模型参数数量
        num_parameters = 0
        parameters = self.model.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        self.logger.info('number of model parameters: {}'.format(num_parameters))

        # 记录参数设置
        self.logger.info("args:{}".format(config))

        self.config = config

    def train_epoch(self, train_dataloader, optimizer, scheduler, epoch):
        self.model.train()
        device = self.config.device
        # pad_id = args.pad_id
        # sep_id = args.sep_id
        ignore_index = self.config.ignore_index
        epoch_start_time = datetime.now()
        total_loss = 0  # 记录下整个epoch的loss的总和

        # epoch_correct_num:每个epoch中,output预测正确的word的数量
        # epoch_total_num: 每个epoch中,output预测的word的总数量
        epoch_correct_num, epoch_total_num = 0, 0

        # 记录验证集的最小loss
        best_val_loss = 10000
        for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
            # 捕获cuda out of memory exception
            try:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = self.model.forward(input_ids, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                loss = loss.mean()

                # 统计该batch的预测token的正确数与总数
                batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
                # 统计该epoch的预测token的正确数与总数
                epoch_correct_num += batch_correct_num
                epoch_total_num += batch_total_num
                # 计算该batch的accuracy
                batch_acc = batch_correct_num / batch_total_num

                total_loss += loss.item()
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps

                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                # 进行一定step的梯度累计之后，更新参数
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # 更新参数
                    optimizer.step()
                    # 更新学习率
                    scheduler.step()
                    # 清空梯度信息
                    optimizer.zero_grad()

                if (batch_idx + 1) % self.config.log_step == 0:
                    self.logger.info("batch {} of epoch {}, loss {}, batch_acc {}, lr {}".format(batch_idx + 1, epoch + 1, loss.item() * self.config.gradient_accumulation_steps, batch_acc, scheduler.get_lr()))
                if (batch_idx + 1) % self.config.save_step == 0:
                    # ========== validate ========== #
                    validate_loss = validate_epoch(model=self.model, validate_dataloader=self.validate_dataloader,
                                                    logger=self.logger, epoch=epoch, args=self.config)

                    # 保存当前困惑度最低的模型，困惑度低，模型的生成效果不一定会越好
                    if validate_loss < best_val_loss:
                        best_val_loss = validate_loss
                        # save model
                        self.logger.info('saving current best model for epoch {} stops {}'.format(epoch + 1, batch_idx+1))
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        model_to_save.save_pretrained(self.model_save_dir)

                del input_ids, outputs

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    self.logger.info("WARNING: ran out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    self.logger.info(str(exception))
                    raise exception

        # 记录当前epoch的平均loss与accuracy
        epoch_mean_loss = total_loss / len(train_dataloader)
        epoch_mean_acc = epoch_correct_num / epoch_total_num
        self.logger.info("epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))
        self.logger.info('epoch {} finished'.format(epoch + 1))
        epoch_finish_time = datetime.now()
        self.logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))

        return epoch_mean_loss


    def train(self):
        train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers, collate_fn=collate_fn,
            drop_last=True
        )
        self.validate_dataloader = DataLoader(self.validate_dataset, batch_size=self.config.batch_size, shuffle=True,
                                         num_workers=self.config.num_workers, collate_fn=collate_fn, drop_last=True)

        t_total = len(train_dataloader) // self.config.gradient_accumulation_steps * self.config.epochs

        optimizer = transformers.AdamW(self.model.parameters(), lr=self.config.lr, eps=self.config.eps)
        # scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=t_total
        )

        self.logger.info('starting training')
        # 开始训练
        for epoch in range(self.config.epochs):
            # ========== train ========== #
            train_loss = self.train_epoch(train_dataloader=train_dataloader, optimizer=optimizer, scheduler=scheduler,epoch=epoch)
            self.logger.info('Epoch {}: training loss {}'.format(epoch, train_loss))

        self.logger.info('training finished')



if __name__ == '__main__':
    kwargs = {"data_dir":os.path.join(os.path.dirname(__file__),'data/chnchat'),
              "model_path":"/Volumes/work/project/unlp/unlp/transformers/gpt2-chnchit"}
    t = Trainer(**kwargs)
    t.train()