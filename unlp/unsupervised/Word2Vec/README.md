# text2vec
1. 包含sentence-bert，腾讯Word2Vec词向量方法；
2. 词向量化表示，句子向量化表示，长文本向量化表示，文本相似度计算。


**Guide**

- [Feature](#Feature)
- [Install](#install)
- [Usage](#usage)
- [Reference](#reference)

# Feature
#### 文本向量表示
- 字词粒度，通过腾讯AI Lab开源的大规模高质量中文[词向量数据（800万中文词轻量版）](https://pan.baidu.com/s/1La4U4XNFe8s5BJqxPQpeiQ) (文件名：light_Tencent_AILab_ChineseEmbedding.bin 密码: tawe），获取字词的word2vec向量表示。
- 句子粒度，通过求句子中所有单词词向量的平均值计算得到。
- 篇章粒度，可以通过gensim库的doc2vec得到，应用较少，本项目不实现。

#### 文本相似度计算

- 基准方法，估计两句子间语义相似度最简单的方法就是求句子中所有单词词向量的平均值，然后计算两句子词向量之间的余弦相似性。
- 词移距离（Word Mover’s Distance），词移距离使用两文本间的词向量，测量其中一文本中的单词在语义空间中移动到另一文本单词所需要的最短距离。

#### query和docs的相似度比较
- rank_bm25方法，使用bm25的变种算法，对query和文档之间的相似度打分，得到docs的rank排序。
- semantic_search方法，使用cosine similarty + topk高效计算，比一对一暴力计算快一个数量级。

## 调研结论

#### 文本相似度计算
- 基准方法（Word2Vec + Cosine）

尽管文本相似度计算的基准方法很简洁，但用平均词向量之间求余弦相似度的表现非常好。实验有以下结论：

    1. 简单word2vec向量比GloVe向量表现的好
    2. 在用word2vec时，尚不清楚使用停用词表或TF-IDF加权是否更有帮助。在STS数据集上，有一点儿帮助；在SICK上没有帮助。
    仅计算未加权的所有word2vec向量平均值表现得很好。
    3. 在使用GloVe时，停用词列表对于达到好的效果非常重要。利用TF-IDF加权没有帮助。

![基准方法效果很好](./docs/base1.jpg)


- 词移距离（WMD）

基于我们的结果，好像没有什么使用词移距离的必要了，因为上述方法表现得已经很好了。只有在STS-TEST数据集上，而且只有在有停止词列表的情况下，词移距离才能和简单基准方法一较高下。

![词移距离的表现令人失望](./docs/move1.jpg)

- 预训练语言模型（SentenceBERT）

对SentenceBERT系列模型经过fine-tune和多语言迁移，可以embedding表示多语言的，长达128个字符的句子。

`paraphrase-multilingual-MiniLM-L12-v2`是`paraphrase-MiniLM-L6-v2`模型的多语言版本，速度快，效果好，支持中文，text2vec默认使用transformers库调用该模型`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。

大家也可以通过sentence-transformers库调用以下SentenceBERT系列模型，具体见[https://github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)


| Model Name | STSb | DupQ | TwitterP | SciDocs | Clustering |  Avg. Performance | Speed |
| :------- | :--------- | :--------- | :---------: | :---------: | :---------: | :---------: | :---------: |
| paraphrase-mpnet-base-v2 | 86.99 | 87.80 | 76.05 | 80.57 | 52.81 | 76.84 | 2800 |
| paraphrase-multilingual-mpnet-base-v2 | 86.82 | 87.50 | 76.52 | 78.66 | 47.46 | 75.39 | 2500 |
| paraphrase-TinyBERT-L6-v2 | 84.91 | 86.93 | 75.39 | 81.51 | 48.04 | 75.36 | 4500 |
| paraphrase-distilroberta-base-v2 | 85.37 | 86.97 | 73.96 | 80.25 | 49.18 | 75.15 | 4000 |
| paraphrase-MiniLM-L12-v2 | 84.41 | 87.28 | 75.34 | 80.08 | 46.95 | 74.81 | 7500 |
| paraphrase-MiniLM-L6-v2 | 84.12 | 87.23 | 76.32 | 78.91 | 45.34 | 74.38 | 14200 |
| paraphrase-multilingual-MiniLM-L12-v2 | 84.42 | 87.52 | 74.94 | 78.27 | 43.87 | 73.80 | 7500 |
| paraphrase-MiniLM-L3-v2 | 82.41 | 88.09 | 76.14 | 77.71 | 43.39 | 73.55 | 19000 |
| distiluse-base-multilingual-cased-v2 | 80.75 | 83.52 | 76.26 | 70.39 | 37.03 | 69.59 | 4000 |
| average_word_embeddings_glove.6B.300d | 61.77 | 78.07 | 68.60 | 63.69 | 30.46 | 60.52 | 34000 |


# Install
```
pip3 install -U text2vec
```

or

```
git clone https://github.com/shibing624/text2vec.git
cd text2vec
python3 setup.py install
```

# Usage

1. 计算文本向量

- 基于`pretrained model`计算文本向量

> `SBert`通过预训练的`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`模型计算句子向量

> `Word2Vec`通过腾讯词向量`Tencent_AILab_ChineseEmbedding.tar.gz`计算各字词的词向量，句子向量通过单词词向量取平均值得到

示例[computing_embeddings.py](../examples/computing_embeddings.py)

```python
import sys

sys.path.append('..')
from Text2Vec import SBert

def compute_emb(model):
    # Embed a list of sentences
    sentences = ['卡',
                 '银行卡',
                 '如何更换花呗绑定银行卡',
                 '花呗更改绑定银行卡',
                 'This framework generates embeddings for each input sentence',
                 'Sentences are passed as a list of string.',
                 'The quick brown fox jumps over the lazy dog.']
    sentence_embeddings = model.encode(sentences)

    print(type(sentence_embeddings), sentence_embeddings.shape)

    # The result is a list of sentence embeddings as numpy arrays
    for sentence, embedding in zip(sentences, sentence_embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")


sbert_model = SBert('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
compute_emb(sbert_model)
```

output:
```
<class 'numpy.ndarray'> (7, 384)
Sentence: 卡
Embedding: [ 1.39491949e-02  8.62287879e-02 -1.35622978e-01 ... ]
Sentence: 银行卡
Embedding: [ 0.06216322  0.2731747  -0.6912158 ... ]
```

返回值`embeddings`是`numpy.ndarray`类型，shape为`(sentence_size, model_embedding_size)`

> `paraphrase-multilingual-MiniLM-L12-v2`是`sentence-bert`预训练模型，Multilingual knowledge distilled version of multilingual 
Universal Sentence Encoder. Supports 50+ languages: Arabic, Chinese, Dutch, English, French, German, Italian, Korean, Polish, 
Portuguese, Russian, Spanish, Turkish.
模型自动下载到本机路径：`~/.cache/torch/sentence_transformers/`

> `w2v-light-tencent-chinese`是`Word2Vec`的轻量版腾讯词向量模型，模型自动下载到本机路径：`~/.text2vec/datasets/light_Tencent_AILab_ChineseEmbedding.bin`



- 预训练词向量模型

以下提供两种`Word2Vec`词向量，任选一个：

  - 轻量版腾讯词向量 [百度云盘-密码:tawe](https://pan.baidu.com/s/1La4U4XNFe8s5BJqxPQpeiQ) 或 [谷歌云盘](https://drive.google.com/u/0/uc?id=1iQo9tBb2NgFOBxx0fA16AZpSgc-bG_Rp&export=download)，二进制，运行程序，自动下载到 `~/.text2vec/datasets/light_Tencent_AILab_ChineseEmbedding.bin`
  - [腾讯词向量-官方全量](https://ai.tencent.com/ailab/nlp/zh/data/Tencent_AILab_ChineseEmbedding.tar.gz), 6.78G放到： `~/.text2vec/datasets/Tencent_AILab_ChineseEmbedding.txt`，腾讯词向量主页：https://ai.tencent.com/ailab/nlp/zh/embedding.html 词向量下载地址：https://ai.tencent.com/ailab/nlp/zh/data/Tencent_AILab_ChineseEmbedding.tar.gz  更多查看[腾讯词向量介绍-wiki](https://github.com/shibing624/text2vec/wiki/%E8%85%BE%E8%AE%AF%E8%AF%8D%E5%90%91%E9%87%8F%E4%BB%8B%E7%BB%8D)



2. 计算句子之间的相似度值

示例[semantic_text_similarity.py](../examples/semantic_text_similarity.py)

```python
import sys

sys.path.append('..')
from Text2Vec import Similarity

# Two lists of sentences
sentences1 = ['如何更换花呗绑定银行卡',
              'The cat sits outside',
              'A man is playing guitar',
              'The new movie is awesome']

sentences2 = ['花呗更改绑定银行卡',
              'The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']

sim_model = Similarity(similarity_type='cosine', embedding_type='sbert')
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        score = sim_model.get_score(sentences1[i], sentences2[j])
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[j], score))
```

output:
```shell
如何更换花呗绑定银行卡 		 花呗更改绑定银行卡 		 Score: 0.9477
如何更换花呗绑定银行卡 		 The dog plays in the garden 		 Score: -0.1748
如何更换花呗绑定银行卡 		 A woman watches TV 		 Score: -0.0839
如何更换花呗绑定银行卡 		 The new movie is so great 		 Score: -0.0044
The cat sits outside 		 花呗更改绑定银行卡 		 Score: -0.0097
The cat sits outside 		 The dog plays in the garden 		 Score: 0.1908
The cat sits outside 		 A woman watches TV 		 Score: -0.0203
The cat sits outside 		 The new movie is so great 		 Score: 0.0302
A man is playing guitar 		 花呗更改绑定银行卡 		 Score: -0.0010
A man is playing guitar 		 The dog plays in the garden 		 Score: 0.1062
A man is playing guitar 		 A woman watches TV 		 Score: 0.0055
A man is playing guitar 		 The new movie is so great 		 Score: 0.0097
The new movie is awesome 		 花呗更改绑定银行卡 		 Score: 0.0302
The new movie is awesome 		 The dog plays in the garden 		 Score: -0.0160
The new movie is awesome 		 A woman watches TV 		 Score: 0.1321
The new movie is awesome 		 The new movie is so great 		 Score: 0.9591
```

> 句子余弦相似度值`score`范围在-1到1之间，值越大越相似。

3. 计算句子与文档集之间的相似度值

一般在文档候选集中找与query最相似的文本，常用于QA场景的问句相似匹配、文本相似检索等任务。


示例[semantic_search.py](../examples/semantic_search.py)

```python
import sys

sys.path.append('..')
from Text2Vec import SBert, cos_sim, semantic_search

embedder = SBert()

# Corpus with example sentences
corpus = [
    '花呗更改绑定银行卡',
    '我什么时候开通了花呗',
    'A man is eating food.',
    'A man is eating a piece of bread.',
    'The girl is carrying a baby.',
    'A man is riding a horse.',
    'A woman is playing violin.',
    'Two men pushed carts through the woods.',
    'A man is riding a white horse on an enclosed ground.',
    'A monkey is playing drums.',
    'A cheetah is running behind its prey.'
]
corpus_embeddings = embedder.encode(corpus)

# Query sentences:
queries = [
    '如何更换花呗绑定银行卡',
    'A man is eating pasta.',
    'Someone in a gorilla costume is playing a set of drums.',
    'A cheetah chases prey on across a field.']

for query in queries:
    query_embedding = embedder.encode(query)
    hits = semantic_search(query_embedding, corpus_embeddings, top_k=5)
    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")
    hits = hits[0]  # Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
```
output:
```shell
Query: 如何更换花呗绑定银行卡
Top 5 most similar sentences in corpus:
花呗更改绑定银行卡 (Score: 0.9477)
我什么时候开通了花呗 (Score: 0.3635)
A man is eating food. (Score: 0.0321)
A man is riding a horse. (Score: 0.0228)
Two men pushed carts through the woods. (Score: 0.0090)
======================
Query: A man is eating pasta.
Top 5 most similar sentences in corpus:
A man is eating food. (Score: 0.6734)
A man is eating a piece of bread. (Score: 0.4269)
A man is riding a horse. (Score: 0.2086)
A man is riding a white horse on an enclosed ground. (Score: 0.1020)
A cheetah is running behind its prey. (Score: 0.0566)
======================
Query: Someone in a gorilla costume is playing a set of drums.
Top 5 most similar sentences in corpus:
A monkey is playing drums. (Score: 0.8167)
A cheetah is running behind its prey. (Score: 0.2720)
A woman is playing violin. (Score: 0.1721)
A man is riding a horse. (Score: 0.1291)
A man is riding a white horse on an enclosed ground. (Score: 0.1213)
======================
Query: A cheetah chases prey on across a field.
Top 5 most similar sentences in corpus:
A cheetah is running behind its prey. (Score: 0.9147)
A monkey is playing drums. (Score: 0.2655)
A man is riding a horse. (Score: 0.1933)
A man is riding a white horse on an enclosed ground. (Score: 0.1733)
A man is eating food. (Score: 0.0329)
```

> `Score`的值越大，表示该query与corpus的相似度越近。


# Reference
- [将句子表示为向量（上）：无监督句子表示学习（sentence embedding）](https://www.cnblogs.com/llhthinker/p/10335164.html)
- [将句子表示为向量（下）：无监督句子表示学习（sentence embedding）](https://www.cnblogs.com/llhthinker/p/10341841.html)
- [A Simple but Tough-to-Beat Baseline for Sentence Embeddings[Sanjeev Arora and Yingyu Liang and Tengyu Ma, 2017]](https://openreview.net/forum?id=SyK00v5xx)
- [四种计算文本相似度的方法对比[Yves Peirsman]](https://zhuanlan.zhihu.com/p/37104535)
- [Improvements to BM25 and Language Models Examined](http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf)
- [CoSENT：比Sentence-BERT更有效的句向量方案](https://kexue.fm/archives/8847)