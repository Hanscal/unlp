# -*- coding: utf-8 -*-
import glob
import json
import csv
import os
import time
import jieba
import copy
import collections

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
import torch
from tqdm import tqdm

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.

# 词汇表大小
VOCAB_SIZE = 50000


def timer(func):
    """耗时装饰器，计算函数运行时长"""

    def wrapper(*args, **kwargs):
        start = time.time()
        r = func(*args, **kwargs)
        end = time.time()
        cost = end - start
        print(f"Cost time: {cost} s")
        return r

    return wrapper

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, text_b):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BuildDataset(object):
    def __init__(self, save_dir):
        for tag in ['train', 'dev', 'test']:
            self._write_txt_json(save_dir, tag, text_path=os.path.join(save_dir, tag+'_text.txt'),
                                 label_path=os.path.join(save_dir, tag+'_label.txt'))

        # 如果没有vocab.txt文件，则需要build
        if not os.path.exists(os.path.join(save_dir, 'vocab.txt')):
            examples = []
            examples.extend(self.get_train_examples(save_dir))
            examples.extend(self.get_dev_examples(save_dir))
            examples.extend(self.get_test_examples(save_dir))
            vocab_counter = collections.Counter()
            for e in tqdm(examples,desc="building vocab"):
                article = e.text_a
                abstract = e.text_b
                art_tokens = article.split(' ')
                abs_tokens = abstract.split(' ')
                abs_tokens = [t for t in abs_tokens if
                              t not in [SENTENCE_START, SENTENCE_END]]  # 从词典中删除这些符号
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]  # 去掉句子开头结尾的空字符
                tokens = [t for t in tokens if t != ""]  # 删除空行
                vocab_counter.update(tokens)
            print("Writing vocab file...")
            with open(os.path.join(save_dir, "vocab.txt"), 'w', encoding='utf-8') as writer:
                for word, count in vocab_counter.most_common(VOCAB_SIZE):
                    writer.write(word + ' ' + str(count) + '\n')
            print("Finished writing vocab file")

    def _write_txt_json(self, save_dir, tag='train', text_path='', label_path=''):
        json_path = os.path.join(save_dir, tag+'.json')
        if not os.path.exists(json_path):
            text_list = []
            label_list = []
            with open(text_path, 'r') as ft:
                for i in tqdm(ft.readlines(),desc="reading text.txt file"):
                    i = i.strip()
                    if not i:
                        continue
                    text_list.append(i)
            with open(label_path, 'r') as fl:
                for i in tqdm(fl.readlines(),desc="reading label.txt file"):
                    i = i.strip()
                    if not i:
                        continue
                    label_list.append(i)
            assert len(text_list) == len(label_list)
            # write to json
            with open(json_path, 'w') as fs:
                for i, j in tqdm(zip(text_list, label_list),desc="writing to json"):
                    i_words = jieba.cut(i.strip())
                    i_word_list = list(i_words)
                    j_words = jieba.cut(j.strip())
                    j_word_list = list(j_words)
                    fs.write(json.dumps({"text_a":' '.join(i_word_list).strip(), "text_b":' '.join(j_word_list).strip()},ensure_ascii=False))
                    fs.write('\n')

    def _read_json(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line.strip())
                text_a = line['text_a']
                text_b = line.get('text_b','')
                lines.append({'text_a':text_a, 'text_b':text_b})
        return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['text_a']
            text_b = line.get('text_b','')
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples


class Vocab(object):
    """
    vocab_file content(word count):
            to 5751035
            a 5100555
            and 4892247
    """

    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: {}'.format(line))
                    continue
                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as {}; we now have {} words. Stopping reading.".format(max_size, self._count))
                    break
        print("Finished constructing vocabulary of {} total words. Last word added: {}".format(self._count, self._id_to_word[self._count-1]))

    def word2id(self, word):
        """获取单个词语的id"""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """根据id解析出对应的词语"""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """获取加上特殊符号后的词汇表数量"""
        return self._count

    def write_metadata(self, fpath):
        print("Writing word embedding metadata file to {}...",format(fpath))
        with open(fpath, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        for i in range(self.size()):
            writer.writerow({"word": self._id_to_word[i]})


def example_generator(data_path, single_pass=None):
    """读取目录下的文件"""
    with open(data_path, 'r') as f:
        for line in f:
            line = json.loads(line.strip())
            text_a = line['text_a']
            text_b = line.get('text_b', '')
            yield {'text_a': text_a, 'text_b': "%s %s %s" % (SENTENCE_START, text_b, SENTENCE_END)}


def article2ids(article_words, vocab):
    """返回两个列表：将文章的词汇转换为id,包含oov词汇id; oov词汇"""
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:         # If w is OOV
            if w not in oovs:   # Add to list of OOVs
                oovs.append(w)
                oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
                ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is an OOV word
            if w in article_oovs: # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w) # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else: # If w is an out-of-article OOV
                ids.append(unk_id) # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)   # might be [UNK]
        except ValueError as e:    # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e: # i doesn't correspond to an article oov
                raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def abstract2sents(abstract):
    """abstract 是用<s></s>分割的多句话，需要分割成n句话的列表并去掉句子分割符号"""
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p+len(SENTENCE_START):end_p])
        except ValueError as e: 
            # no more sentences
            return sents


def show_art_oovs(article, vocab):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token: # w is oov
            if article_oovs is None: # baseline mode
                new_words.append("__%s__" % w)
            else: 
                # pointer-generator mode
                pass
            if w in article_oovs:
                new_words.append("__%s__" % w)
            else:
                new_words.append("!!__%s__!!" % w)
        else: # w is in-vocab word
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str


if __name__ == '__main__':
    # 通过文件得到我们需要的训练数据
    p = BuildDataset('/Volumes/work/project/unlp/unlp/supervised/nlg/data/weibo/data')


