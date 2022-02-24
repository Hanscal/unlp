import sys
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

sys.path.append(os.path.dirname(__file__))
from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger

from nmodels.transformer import WEIGHTS_NAME, BertConfig, AlbertConfig
from nmodels.bert_for_ner import BertCrfForNer
from nmodels.albert_for_ner import AlbertCrfForNer
from processors.utils_ner import CNerTokenizer, get_entities
from processors.ner_seq import convert_examples_to_features
from processors.ner_seq import ner_processors as processors
from tools.config import get_argparse

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert-base-chinese': (BertConfig, BertCrfForNer, CNerTokenizer),
    'chinese-bert-wwm-ext': (BertConfig, BertCrfForNer, CNerTokenizer),
    'ernie-1.0': (BertConfig, BertCrfForNer, CNerTokenizer),
    'albert-base-chinese': (AlbertConfig, AlbertCrfForNer, CNerTokenizer),
}

class NERPredictor(object):
    def __init__(self, **kwargs):
        self.args = get_argparse().parse_args()
        # 利用kwargs来更新args
        args_bak = vars(self.args)
        for k, v in kwargs.items():
            if k in args_bak:
                args_bak[k] = v

        # Setup CUDA, GPU
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.n_gpu = torch.cuda.device_count()

        # Set seed
        seed_everything(self.args.seed)

        # Prepare NER task
        self.args.task_name = 'cluener'
        self.processor = processors[self.args.task_name]()
        self.args.data_dir = os.path.join(self.args.data_dir, 'data') if not self.args.data_dir.endswith('data') else self.args.data_dir
        self.label_list = self.processor.get_labels(self.args.data_dir, self.args.markup)
        self.args.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.args.label2id = {label: i for i, label in enumerate(self.label_list)}
        num_labels = len(self.label_list)

        # Load model and tokenizer
        self.args.model_type = self.args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args.model_type]

        print("loading model from {}".format(self.args.model_name_or_path))
        self.config = config_class.from_pretrained(self.args.model_name_or_path, num_labels=num_labels)
        self.model = model_class.from_pretrained(self.args.model_name_or_path, config=self.config)
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        self.model.to(self.args.device)
        self.model.eval()
        self.tokenizer = tokenizer_class.from_pretrained(self.args.model_name_or_path, do_lower_case=self.args.do_lower_case)


    def predict(self, texts:list):
        b0 = time.time()
        test_dataset = self.load_examples(texts)
        logger.info("***** Running prediction %s *****")
        batch = tuple(t.to(self.args.device) for t in test_dataset.tensors)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None, 'input_lens': batch[4]}
            if self.args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if self.args.model_type in ["bert", "xlnet"] else None)
            outputs = self.model(**inputs)
            logits = outputs[0]
            tags = self.model.crf.decode(logits, inputs['attention_mask'])
            tags = tags.squeeze(0).cpu().numpy().tolist()
        res = []
        for tag, text in zip(tags, texts):
            preds = tag[1:-1]  # [CLS]XXXX[SEP]
            label_entities = get_entities(preds, self.args.id2label, self.args.markup)
            json_d = {}
            json_d['tag_seq'] = " ".join([self.args.id2label[x] for x in preds])
            json_d['entities'] = label_entities
            json_d['label'] = {}
            words = list(text)
            if len(label_entities) != 0:
                for subject in label_entities:
                    tag = subject[0]
                    start = subject[1]
                    end = subject[2]
                    word = "".join(words[start:end + 1])
                    if tag in json_d['label']:
                        if word in json_d['label'][tag]:
                            json_d['label'][tag][word].append([start, end])
                        else:
                            json_d['label'][tag][word] = [[start, end]]
                    else:
                        json_d['label'][tag] = {}
                        json_d['label'][tag][word] = [[start, end]]
            res.append(json_d)
            print("predict single costs {:.2f}s".format(time.time() - b0))
        return res

    def load_examples(self, texts:list):
        # Load data features from cache or dataset file
        examples = self.processor.get_single_example(texts=texts)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=self.tokenizer,
                                                label_list=self.label_list,
                                                max_seq_length=self.args.train_max_seq_length,
                                                cls_token_at_end=bool(self.args.model_type in ["xlnet"]),
                                                pad_on_left=bool(self.args.model_type in ['xlnet']),
                                                cls_token=self.tokenizer.cls_token,
                                                cls_token_segment_id=2 if self.args.model_type in ["xlnet"] else 0,
                                                sep_token=self.tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0,
                                                )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
        return dataset

if __name__ == "__main__":
    text = ['在豪门被多线作战拖累时，正是他们悄悄追赶上来的大好时机。重新找回全队的凝聚力是拉科赢球的资本。',"不久后，“星展中国”南宁分行也将择机开业。除新加坡星展银行外，多家外资银行最近纷纷扩大在华投资，"]
    kwargs = {"model_name_or_path": "/Volumes/work/project/unlp/unlp/supervised/ner/data/cluner/saved_dict/bert-base-chinese",
        "data_dir": "/Volumes/work/project/unlp/unlp/supervised/ner/data/cluner", "model_type": "bert-base-chinese"}
    p = NERPredictor(**kwargs)
    res = p.predict(texts=text)
    print(res)
