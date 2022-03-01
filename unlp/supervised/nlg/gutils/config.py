# -*- coding: utf-8 -*-

"""
@Time    : 2022/2/24 11:48 上午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""

import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    # Required parameters
    # 需要根据输入来进行确定
    parser.add_argument("--model_type", default='point-net', type=str, help="The model type supported")
    parser.add_argument("--data_dir", default='../data/weibo', type=str,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.", )
    parser.add_argument("--model_path", default='', type=str, help="Path to pre-trained model or shortcut name selected in the list: ")

    # Other parameters
    parser.add_argument('--hidden_dim', default=256, type=int, help="model number of hidden units")
    parser.add_argument('--emb_dim', default=128, type=int, help="model embedding dim")
    parser.add_argument("--max_enc_steps", default=400, type=int, help="The maximum total input time steps after tokenization. Sequences longer "
                             "than this will be truncated")
    parser.add_argument("--max_dec_steps", default=40, type=int, help="max decode time steps", )
    parser.add_argument("--min_dec_steps", default=20, type=int, help="min decode time steps", )
    parser.add_argument("--beam_size", default=4, type=int, help="beam size for decode", )
    parser.add_argument("--vocab_size", default=50000, type=int,  help="", )
    parser.add_argument("--pointer_gen", default=True, type=bool, help="use generate network")
    parser.add_argument("--is_coverage", default=True, type=bool, help="")
    parser.add_argument("--cov_loss_wt", default=1.0, type=float, help="")

    # training
    parser.add_argument("--lr", default=0.15, type=float,  help="Whether to adversarial training.")
    parser.add_argument("--lr_coverage", default=0.15, type=float, help="Whether to adversarial training.")
    parser.add_argument('--adagrad_init_acc', default=0.1, type=float, help="Epsilon for adversarial.")
    parser.add_argument('--rand_unif_init_mag', default=0.02, type=float, help="name for adversarial layer.")
    parser.add_argument("--trunc_norm_init_std", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--num_train_epochs", default=10, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_iter_steps_epoch", default=100000, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
    parser.add_argument("--eps", default=1e-12, type=float, help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed", type=int, default=1234, help="random seed for initialization")
    args = parser.parse_args()
    return args

def set_gpt_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='dialog-gpt', type=str, help="The model type supported")
    parser.add_argument('--data_dir', default='../data/chnchat', type=str, required=False, help='训练集路径')
    parser.add_argument('--model_path', default='', type=str, required=False, help='预训练的模型的路径')

    parser.add_argument('--log', default=True, help="是否记录日志")
    parser.add_argument('--ignore_index', default=-100, type=int, required=False, help='对于ignore_index的label token不计算梯度')

    # train 用到参数
    parser.add_argument('--epochs', default=100, type=int, required=False, help='训练的最大轮次')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='训练的batch size')

    parser.add_argument('--lr', default=2.6e-5, type=float, required=False, help='学习率')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='衰减率')
    parser.add_argument('--log_step', default=10, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument("--save_step", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, required=False)
    parser.add_argument('--max_len', default=150, type=int, required=False, help='训练时，输入数据的最大长度')

    parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=4, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--patience', type=int, default=0, help="用于early stopping,设为0时,不进行early stopping.early stop得到的模型的生成效果不一定会更好。")
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warm up步数')

    # predict 用到参数
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--max_decode_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=3, help="dialogue history的最大长度")

    args = parser.parse_args()
    args_bak = vars(args)
    for k, v in kwargs.items():
        if k in args_bak:
            args_bak[k] = v
    return args
