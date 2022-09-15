# -*- coding: utf-8 -*-

"""
@Time    : 2022/2/14 5:52 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from robust import MultiFocalLoss

WEIGHTS_NAME = 'pytorch_model.bin'

class Evaluate(object):
    def __init__(self):
        self.FocalLoss = MultiFocalLoss()

    def test(self, config, model, test_iter):
        # test
        save_path = os.path.join(config.save_path, WEIGHTS_NAME)
        model.load_state_dict(torch.load(save_path))
        model.eval()
        start_time = time.time()
        test_acc, test_loss, test_report, test_confusion = self.evaluate(config, model, test_iter, test=True)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_acc))
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)
        print("Time usage:", time.time() - start_time)
        return test_acc, test_loss, test_report, test_confusion

    def evaluate(self, config, model, data_iter, test=False):
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for texts, labels in data_iter:
                outputs = model(texts)
                # loss = F.cross_entropy(outputs, labels)
                # 改成focal loss
                loss = self.FocalLoss(outputs, labels, len(config.class_list))
                loss_total += loss
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)

        acc = metrics.accuracy_score(labels_all, predict_all)
        if test:
            report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            return acc, loss_total / len(data_iter), report, confusion
        return acc, loss_total / len(data_iter)