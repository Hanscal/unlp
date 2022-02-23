# -*- coding: utf-8 -*-

"""
@Time    : 2022/2/14 5:52 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(__file__))
from evaluate import Evaluate

WEIGHTS_NAME = 'pytorch_model.bin'

class Train(object):
    def __init__(self):
        self.eval = Evaluate()

    def train(self, config, model, train_iter, dev_iter, test_iter):
        start_time = time.time()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        total_batch = 0  # 记录进行到多少batch
        dev_best_loss = float('inf')
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升
        writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
        for epoch in range(config.num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
            # scheduler.step() # 学习率衰减
            for i, (trains, labels) in enumerate(train_iter):
                outputs = model(trains)
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                if total_batch % 100 == 0:
                    # 每多少轮输出在训练集和验证集上的效果
                    true = labels.data.cpu()
                    predic = torch.max(outputs.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, predic)
                    dev_acc, dev_loss = self.eval.evaluate(config, model, dev_iter)
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        save_path = config.save_path
                        if config.model_name in ["BERT", "ERNIE"]:
                            # If we save using the predefined names, we can load using `from_pretrained`
                            save_path = os.path.join(config.save_path, WEIGHTS_NAME)
                        torch.save(model.state_dict(), save_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = time.time() - start_time
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                    writer.add_scalar("loss/train", loss.item(), total_batch)
                    writer.add_scalar("loss/dev", dev_loss, total_batch)
                    writer.add_scalar("acc/train", train_acc, total_batch)
                    writer.add_scalar("acc/dev", dev_acc, total_batch)
                    model.train()
                total_batch += 1
                if total_batch - last_improve > config.require_improvement:
                    # 验证集loss超过1000batch没下降，结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break
        writer.close()
        self.eval.test(config, model, test_iter)


class TrainTransfomer(object):
    def __init__(self):
        from models.transformer.bert_optimization import BertAdam
        self.BertAdam = BertAdam
        self.eval = Evaluate()

    def init_network(self, model, method='xavier', exclude='embedding', seed=123):
        for name, w in model.named_parameters():
            if exclude not in name:
                if len(w.size()) < 2:
                    continue
                if 'weight' in name:
                    if method == 'xavier':
                        nn.init.xavier_normal_(w)
                    elif method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass

    def train(self, config, model, train_iter, dev_iter, test_iter):
        start_time = time.time()
        model.train()
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        optimizer = self.BertAdam(optimizer_grouped_parameters,
                             lr=config.learning_rate,
                             warmup=0.05,
                             t_total=len(train_iter) * config.num_epochs)
        total_batch = 0  # 记录进行到多少batch
        dev_best_loss = float('inf')
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升
        model.train()
        for epoch in range(config.num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
            for i, (trains, labels) in enumerate(train_iter):
                outputs = model(trains)
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                if total_batch % 100 == 0:
                    # 每多少轮输出在训练集和验证集上的效果
                    true = labels.data.cpu()
                    predic = torch.max(outputs.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, predic)
                    dev_acc, dev_loss = self.eval.evaluate(config, model, dev_iter)
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        save_path = config.save_path
                        if config.model_name in ["BERT", "ERNIE"]:
                            # If we save using the predefined names, we can load using `from_pretrained`
                            save_path = os.path.join(config.save_path, WEIGHTS_NAME)
                        torch.save(model.state_dict(), save_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = time.time() - start_time
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                    model.train()
                total_batch += 1
                if total_batch - last_improve > config.require_improvement:
                    # 验证集loss超过1000batch没下降，结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break
        self.eval.test(config, model, test_iter)
