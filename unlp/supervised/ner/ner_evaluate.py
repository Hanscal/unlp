import sys
import os

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(__file__))
from callback.progressbar import ProgressBar
from nmodels.transformer import WEIGHTS_NAME, BertConfig, AlbertConfig
from nmodels.bert_for_ner import BertCrfForNer
from nmodels.albert_for_ner import AlbertCrfForNer
from processors.utils_ner import CNerTokenizer, get_entities
from processors.ner_seq import ner_processors as processors

from metrics.ner_metrics import SeqEntityScore

from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger
from tools.config import get_argparse

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert-base-chinese': (BertConfig, BertCrfForNer, CNerTokenizer),
    'chinese-bert-wwm-ext': (BertConfig, BertCrfForNer, CNerTokenizer),
    'ernie-1.0': (BertConfig, BertCrfForNer, CNerTokenizer),
    'albert-base-chinese': (AlbertConfig, AlbertCrfForNer, CNerTokenizer),
}

class NEREvaluator(object):
    def __init__(self, **kwargs):
        args = get_argparse().parse_args()
        # 利用kwargs来更新args
        args_bak = vars(args)
        for k, v in kwargs.items():
            if k in args_bak:
                args_bak[k] = v
        print(args)

        # Setup CUDA, GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.device = device

        # Set seed
        seed_everything(args.seed)

        # Prepare NER task
        args.task_name = "cluener" # 固定
        processor = processors[args.task_name]()
        args.data_dir = os.path.join(args.data_dir, 'data') if not args.data_dir.endswith('data') else args.data_dir
        label_list = processor.get_labels(args.data_dir, args.markup)
        args.id2label = {i: label for i, label in enumerate(label_list)}
        args.label2id = {label: i for i, label in enumerate(label_list)}

        self.args = args

    def evaluate(self, model, eval_dataloader, prefix=""):
        logger.info("Evaluation parameters %s", self.args)
        metric = SeqEntityScore(self.args.id2label, markup=self.args.markup)

        # Eval!
        logger.info("***** Running evaluation %s *****", prefix)
        eval_loss = 0.0
        nb_eval_steps = 0
        pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
        if isinstance(model, nn.DataParallel):
            model = model.module
        model.eval()

        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], 'input_lens': batch[4]}
                inputs["token_type_ids"] = batch[2]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                tags = model.crf.decode(logits, inputs['attention_mask'])
            if self.args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            out_label_ids = inputs['labels'].cpu().numpy().tolist()
            input_lens = inputs['input_lens'].cpu().numpy().tolist()
            tags = tags.squeeze(0).cpu().numpy().tolist()
            for i, label in enumerate(out_label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif j == input_lens[i] - 1:
                        metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                        break
                    else:
                        temp_1.append(self.args.id2label[out_label_ids[i][j]])
                        temp_2.append(self.args.id2label[tags[i][j]])
            pbar(step)
        logger.info("\n")
        eval_loss = eval_loss / nb_eval_steps
        eval_info, entity_info = metric.result()
        results = {f'{key}': value for key, value in eval_info.items()}
        results['loss'] = eval_loss
        logger.info("***** Eval results %s *****", prefix)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
        logger.info(info)
        logger.info("***** Entity results %s *****", prefix)
        for key in sorted(entity_info.keys()):
            logger.info("******* %s results ********" % key)
            info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
            logger.info(info)
        return results


if __name__ == "__main__":
    pass