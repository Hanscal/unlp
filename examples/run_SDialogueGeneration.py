# -*- coding: utf-8 -*-

"""
@Time    : 2022/3/7 5:13 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""

from unlp import SDialogueGeneration


def summarize(model_path, model_type, mode, datadir, text_list: list = []):
    # 现在只支持"point-net"这个模型来做文本摘要任务
    model = SDialogueGeneration(model_path=model_path, model_type=model_type, mode=mode, datadir=datadir)
    res = model.run(text=text_list)
    return res


if __name__ == '__main__':
    datadir = "/Volumes/work/project/unlp/unlp/supervised/nlg/data/weibo"
    model_path = "/Volumes/work/project/unlp/unlp/supervised/nlg/data/weibo/saved_dict/point-net/point-net.pt"
    res = summarize(model_path=model_path, model_type='point-net', mode='train', datadir=datadir)
    print(res)