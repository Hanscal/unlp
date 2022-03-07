# 无监督学习

# 概述
    - 无监督学习中有关键词抽取，向量嵌入和相似度计算等方法；
    - 目前支持的功能有文本语义搜索，文本相似度计算，文本向量表征和关键词抽取。


# 语义搜索
    - 1.sentence bert向量表征后的语义检索。
    - 2.word2vec向量表征后的语义检索。
    - 3.bm25检索算法

# 文本相似度计算
    - 1.sentence bert向量表征后的相似度计算
    - 2.word2vec向量表征后的相似度计算

# 文本向量表征
    - 1.sentence bert向量表征
    - 2.word2vec向量表征

# 关键词抽取
    - 1.keyword bert算法
    - 2.tf-idf算法

# 使用方法
### 目前支持
### 1. 关键词抽取  
    - **通过model_path和model_type来制定模型**   
    -   model_path可以为相应的model权重，如果为''，则会自动下载model权重；  
    -   model_type目前支持['keybert','tfidf']   
    - **text：传入参数为需要提取关键词的句子列表**  

```py
from unlp import UKeywordsExtract
model = UKeywordsExtract(model_path, model_type)
keywords = model.run(text)  # text为list, 支持批量抽取
```

### 2. 获得嵌入向量  
    - **通过model_path和model_type来制定模型**  
    -   model_path可以为相应的model权重，如果为''，则会自动下载model权重；  
    -   model_type目前支持['sentbert', 'w2v'] 
    - **text：传入参数为需要求向量的句子或者词的列表**

```py
from unlp import UTextEmbedding
model = UTextEmbedding(model_path, model_type)
embeddings = model.run(text)  # text为list, 支持批量文本嵌入向量
```

### 3. 获得文本相似度  
    - **通过model_path和model_type来制定模型**  
    -   model_path可以为相应的model权重，如果为''，则会自动下载model权重；  
    -   model_type目前支持['sentbert', 'w2v'] 
    - **text：传入参数为需要求向量的句子或者词的列表** 

```py
from unlp import UTextSimilarity
model = UTextSimilarity(model_path, model_type)
similarity = model.run(texta, textb)  # texta和textb为str, 实现文本语义相似度计算
```

### 4. 文本语义检索  
    - **通过model_path和model_type来制定模型**  
    -   model_path可以为相应的model权重，如果为''，则会自动下载model权重；  
    -   model_type目前支持['bm25', 'sentbert', 'w2v']  
    -   corpus为搜索的库，需要在实例化类的时候进行向量嵌入，为list格式
    - **queries：需要查询句子或者词的列表**  
    - **top_k: 每个查询返回的条数，默认是5条；   

```py
from unlp import USemanticSearch
model = USemanticSearch(model_path, model_type, corpus)
retrieves = model.run(queries, top_k)  # queries为list, 实现批量文本语义搜索
```

   
# 相关论文  
[1] keyword-bert: Self-Supervised Contextual Keyword and Keyphrase Retrieval with Self-Labelling.