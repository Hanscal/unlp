import sys
import os
import time
import shutil

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

sys.path.append(os.path.dirname(__file__))
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar

from nmodels.transformer import WEIGHTS_NAME, BertConfig, AlbertConfig
from nmodels.bert_for_ner import BertCrfForNer
from nmodels.albert_for_ner import AlbertCrfForNer
from processors.utils_ner import CNerTokenizer, get_entities
from processors.ner_seq import ner_processors as processors
from processors.ner_seq import collate_fn
from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger
from tools.dutils import load_and_cache_examples
from tools.get_file import get_file
from tools.config import get_argparse

from ner_evaluate import NEREvaluator

USER_DATA_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_NAME = 'config.json'

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert-base-chinese': (BertConfig, BertCrfForNer, CNerTokenizer),
    'chinese-bert-wwm-ext': (BertConfig, BertCrfForNer, CNerTokenizer),
    'ernie-1.0': (BertConfig, BertCrfForNer, CNerTokenizer),
    'albert-base-chinese': (AlbertConfig, AlbertCrfForNer, CNerTokenizer),
}

class NERTrainer(object):
    model_key_map = {
        # 谷歌预训练模型
        'bert-base-chinese': {
            'model_url': "https://huggingface.co/bert-base-chinese/resolve/main/pytorch_model.bin",
            'config_url': "https://huggingface.co/bert-base-chinese/resolve/main/config.json",
            'vocab_url': "https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt"
        },
        # 哈工大预训练模型
        'chinese-bert-wwm-ext': {
            'model_url': "https://huggingface.co/hfl/chinese-bert-wwm-ext/resolve/main/pytorch_model.bin",
            'config_url': "https://huggingface.co/hfl/chinese-bert-wwm-ext/resolve/main/config.json",
            'vocab_url': "https://huggingface.co/hfl/chinese-bert-wwm-ext/resolve/main/vocab.txt"
        },
        # 百度ernie模型
        'ernie-1.0': {
            'model_url': "https://huggingface.co/hfl/chinese-bert-wwm-ext/resolve/main/pytorch_model.bin",
            'config_url': "https://huggingface.co/hfl/chinese-bert-wwm-ext/resolve/main/config.json",
            'vocab_url': "https://huggingface.co/hfl/chinese-bert-wwm-ext/resolve/main/vocab.txt"
        },
        # 快速轻量化albert模型
        'albert-base-chinese': {
            'model_url': "https://huggingface.co/voidful/albert_chinese_base/resolve/main/pytorch_model.bin",
            'config_url': "https://huggingface.co/voidful/albert_chinese_base/resolve/main/config.json",
            'vocab_url': "https://huggingface.co/voidful/albert_chinese_base/resolve/main/vocab.txt"
        }
    }

    def __init__(self, **kwargs):
        args = get_argparse().parse_args()
        # 利用kwargs来更新args
        args_bak = vars(args)
        for k, v in kwargs.items():
            if k in args_bak:
                args_bak[k] = v

        # save dir
        args.model_type = args.model_type # 这个datadir只需给到数据的上两级目录
        # log dir
        log_dir = os.path.join(args.data_dir, 'log', args.model_type)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # model_save_dir
        model_save_dir = os.path.join(args.data_dir, 'saved_dict', args.model_type)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        args_bak['model_save_dir'] = model_save_dir
        # data_dir
        data_dir = os.path.join(args.data_dir, 'data')
        args_bak['data_dir'] = data_dir

        time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        init_logger(log_file=os.path.join(log_dir, f'{args.model_type}-{time_}.log'))

        # Setup CUDA, GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args_bak['n_gpu'] = torch.cuda.device_count()
        args_bak['device'] = device

        # Set seed
        seed_everything(args.seed)

        # Prepare NER task, 目前支持cluener这个数据集
        args_bak['task_name'] = 'cluener'
        processor = processors[args.task_name]()
        label_list = processor.get_labels(data_dir=data_dir, markup=args.markup)
        args.id2label = {i: label for i, label in enumerate(label_list)}
        args.label2id = {label: i for i, label in enumerate(label_list)}
        num_labels = len(label_list)

        args.model_type = args.model_type.lower()
        if args.model_type not in list(MODEL_CLASSES.keys()):
            print("model type should in {}".format(list(MODEL_CLASSES.keys())))
            os._exit(-1)
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

        # 需要通过model_name判断模型是否存在,如果不存在需要自动下载模型
        args.model_name_or_path = os.path.join(USER_DATA_DIR, args.model_name_or_path) if not os.path.exists(args.model_name_or_path) else args.model_name_or_path
        if os.path.exists(args.model_name_or_path) and os.listdir(args.model_name_or_path):
            args.model_name_or_path = args.model_name_or_path
        else:
            get_file(origin=list(self.model_key_map[args.model_name_or_path].values()), extract=False, untar=False,
                     cache_dir=USER_DATA_DIR, cache_subdir=args.model_name_or_path, verbose=1)
            args.model_name_or_path = os.path.join(USER_DATA_DIR, args.model_name_or_path)

        config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels)

        # save config.json to model_save_dir
        shutil.copyfile(os.path.join(args.model_name_or_path, CONFIG_NAME), os.path.join(args.model_save_dir, CONFIG_NAME))
        self.tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        self.model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                            config=config)
        if kwargs.get('resume', ''):
            resume_model_path = os.path.join(args.model_name_or_path, WEIGHTS_NAME)
            print("resume model from {}".format(resume_model_path))
            self.model.load_state_dict(torch.load(resume_model_path, map_location=torch.device('cpu')))

        self.model.to(args.device)
        logger.info(args)
        self.args = args
        self.eval = NEREvaluator(**vars(args))

    def train(self):
        """ Train the model """
        logger.info("Training/evaluation parameters %s", self.args)

        # Load train and dev dataset
        train_dataset = load_and_cache_examples(self.args, self.args.task_name, self.tokenizer, data_type='train')
        eval_dataset = load_and_cache_examples(self.args, self.args.task_name, self.tokenizer, data_type='dev')
        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)

        # 如果n_gpu大于0，多卡训练
        self.args.train_batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size, collate_fn=collate_fn)
        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, collate_fn=collate_fn)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        bert_param_optimizer = list(self.model.bert.named_parameters())
        crf_param_optimizer = list(self.model.crf.named_parameters())
        linear_param_optimizer = list(self.model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay, 'lr': self.args.learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.args.learning_rate},

            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay, 'lr': self.args.crf_learning_rate},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.args.crf_learning_rate},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay, 'lr': self.args.crf_learning_rate},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.args.crf_learning_rate}
        ]
        self.args.warmup_steps = int(t_total * self.args.warmup_proportion)
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    self.args.train_batch_size
                    * self.args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if os.path.exists(self.args.model_name_or_path) and "checkpoint" in self.args.model_name_or_path:
            # set global_step to gobal_step of last saved checkpoint from model path
            global_step = int(self.args.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // self.args.gradient_accumulation_steps)
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        tr_loss, logging_loss = 0.0, 0.0
        f1 = 0
        self.model.zero_grad()
        seed_everything(self.args.seed)  # Added here for reproductibility (even between python 2 and 3)
        for _ in range(int(self.args.num_train_epochs)):
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training/Epoch{}'.format(_))
            for step, batch in enumerate(train_dataloader):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                self.model.train()
                batch = tuple(t.to(self.args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], 'input_lens': batch[4]}
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                pbar(step, {'loss': loss.item()})
                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    scheduler.step()  # Update learning rate schedule
                    optimizer.step()
                    self.model.zero_grad()
                    global_step += 1
                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        # Log metrics, 用acc, recall和f1值
                        eval_res = self.eval.evaluate(self.model, eval_dataloader)
                        # 根据评估结果save model, 只保存最好结果，根据f1来存模型
                        if eval_res['f1'] > f1:
                            f1 = eval_res['f1']
                            # Save model checkpoint
                            model_to_save = (self.model.module if hasattr(self.model, "module") else self.model)  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(self.args.model_save_dir)
                            logger.info("Saving model checkpoint to %s", self.args.model_save_dir)
                            self.tokenizer.save_vocabulary(self.args.model_save_dir) # save vocab.txt
                            logger.info("Saving optimizer and scheduler states to %s", self.args.model_save_dir)
            logger.info("\n")
            if 'cuda' in str(self.args.device):
                torch.cuda.empty_cache()
        return global_step, tr_loss / global_step


if __name__ == "__main__":
    kwargs = {"data_dir":"/Volumes/work/project/unlp/unlp/supervised/ner/data/cluner", "model_type":"bert-base-chinese",
              "model_name_or_path":"/Volumes/work/project/unlp/unlp/transformers/bert-base-chinese"}
    train = NERTrainer(**kwargs)
    res = train.train()
    print(res)
