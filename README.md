# 什么是 unlp

[GitHub欢迎提pr，如果有bug或新需求，请反馈issue](https://github.com/Hanscal/unlp/issues)

unlp是一些经常需要用到的NLP算法包，有助于您学习和使用基于深度学习的文本处理。

### 安装

```py
pip3 install unlp
```

## 使用 unlp
----
1. 主要分为无监督学习和有监督学习的方法；
2. 主要是根据nlp的任务来构建这个包，
比如无监督学习中有关键词抽取，向量嵌入和相似度计算；
监督学习中有分类任务，命名实体识别，文本生成等。

## 无监督学习方法
### 目前支持
### 1. 关键词抽取  
**model：通过model_path和model_type来制定模型**   
  model_path可以为相应的model权重，如果为''，则会自动下载model权重；  
  model_type目前支持['keybert','tfidf']   
**text：传入参数为需要提取关键词的句子列表**  

```py
from unlp import UKeywordsExtract
model = UKeywordsExtract(modelpath, model_type)
keywords = model.run(text)  # text为list, 支持批量抽取
```

### 2. 获得嵌入向量  
**model：通过model_path和model_type来制定模型**  
  model_path可以为相应的model权重，如果为''，则会自动下载model权重；  
  model_type目前支持['sentbert', 'w2v'] 
**text：传入参数为需要求向量的句子或者词的列表**

```py
from unlp import UTextEmbedding
model = UTextEmbedding(modelpath, model_type)
embeddings = model.run(text)  # text为list, 支持批量文本嵌入向量
```

### 3. 获得文本相似度  
**model：通过model_path和model_type来制定模型**  
  model_path可以为相应的model权重，如果为''，则会自动下载model权重；  
  model_type目前支持['sentbert', 'w2v'] 
**text：传入参数为需要求向量的句子或者词的列表** 

```py
from unlp import UTextSimilarity
model = UTextSimilarity(modelpath, model_type)
similarity = model.run(texta, textb)  # texta和textb为str, 实现文本语义相似度计算
```

### 4. 文本语义检索  
**model：通过model_path和model_type来制定模型**  
  model_path可以为相应的model权重，如果为''，则会自动下载model权重；  
  model_type目前支持['bm25', 'sentbert', 'w2v']  
  corpus为搜索的库，需要在实例化类的时候进行向量嵌入，为list格式
**queries：需要查询句子或者词的列表**  
**top_k: 每个查询返回的条数，默认是5条；   

```py
from unlp import USemanticSearch
model = USemanticSearch(modelpath, model_type, corpus)
retrieves = model.run(queries, top_k)  # queries为list, 实现批量文本语义搜索
```

## TODO 监督学习方法
### 1. 文本分类  

### 2. 序列标注  

### 3. 文本生成  

### 4. 文本对相关  
 
