# 什么是 unlp

[GitHub欢迎提pr，如果有bug或新需求，请反馈issue](https://github.com/Hanscal/unlp/issues)

unlp是一些经常需要用到的NLP算法包，有助于您学习和使用基于深度学习的文本处理。

**python3.6+**
### 安装

```py
pip install unlp 
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
**通过model_path和model_type来制定模型**   
  model_path可以为相应的model权重，如果为''，则会自动下载model权重；  
  model_type目前支持['keybert','tfidf']   
**text：传入参数为需要提取关键词的句子列表**  

```py
from unlp import UKeywordsExtract
model = UKeywordsExtract(model_path, model_type)
keywords = model.run(text)  # text为list, 支持批量抽取
```

### 2. 获得嵌入向量  
**通过model_path和model_type来制定模型**  
  model_path可以为相应的model权重，如果为''，则会自动下载model权重；  
  model_type目前支持['sentbert', 'w2v'] 
**text：传入参数为需要求向量的句子或者词的列表**

```py
from unlp import UTextEmbedding
model = UTextEmbedding(model_path, model_type)
embeddings = model.run(text)  # text为list, 支持批量文本嵌入向量
```

### 3. 获得文本相似度  
**通过model_path和model_type来制定模型**  
  model_path可以为相应的model权重，如果为''，则会自动下载model权重；  
  model_type目前支持['sentbert', 'w2v'] 
**text：传入参数为需要求向量的句子或者词的列表** 

```py
from unlp import UTextSimilarity
model = UTextSimilarity(model_path, model_type)
similarity = model.run(texta, textb)  # texta和textb为str, 实现文本语义相似度计算
```

### 4. 文本语义检索  
**通过model_path和model_type来制定模型**  
  model_path可以为相应的model权重，如果为''，则会自动下载model权重；  
  model_type目前支持['bm25', 'sentbert', 'w2v']  
  corpus为搜索的库，需要在实例化类的时候进行向量嵌入，为list格式
**queries：需要查询句子或者词的列表**  
**top_k: 每个查询返回的条数，默认是5条；   

```py
from unlp import USemanticSearch
model = USemanticSearch(model_path, model_type, corpus)
retrieves = model.run(queries, top_k)  # queries为list, 实现批量文本语义搜索
```

## TODO 监督学习方法
### 1. 文本分类  
**通过model_path和model_type来制定模型**  
  model_path可以为相应的model名称:支持['bert-base-chinese','ernie-1.0']或者是模型路径，如果为''，则会自动下载bert-base-chinese权重；  
  model_type目前支持['DPCNN', "FastText", "TextCNN", "TextRCNN", "TextRNN", "TextRNN_Att", "BERT", "ERNIE"]  
  mode为模型的三种模式：['train', "evaluate", "predict"]，分别对应于训练，评估和预测。
  datadir为模型的输入数据，格式可以通过这个命令查看：  
```py
from unlp import ClassificationDataFormat
```
**kwargs：额外需要传入的参数**  
如果是预测predict, run的参数需要传入text=List[str]这样的格式；   
如果是训练train,可以设置resume为True (bool类型）控制是否继续训练，其他预测predict和评估evaluate阶段可以不传入这个参数  

```py
from unlp import STextClassification
model = STextClassification(model_path, model_type, mode, datadir, **kwargs)
res = model.run()  # 实现模型的训练，评估和预测
```
finetune训练代码示例:['BERT',"ERNIE"]需要传入model_path（为预训练模型所在目录或者通过字符串指定下载）, 其他model_type不需要传入，model_type=''

```py
from unlp import STextClassification
model = STextClassification(model_path='bert-base-chinese', model_type='BERT', mode='train', datadir='./data/THUCNews', 
**kwargs)
res = model.run()
```

resume训练代码示例:所有model_type都需要传入model_path,['BERT',"ERNIE"]为保存模型所在目录，其他为模型文件

```py
from unlp import STextClassification
model = STextClassification(model_path='bert-base-chinese', model_type='BERT', mode='train', datadir='./data/THUCNews', 
**{"resume":True})
res = model.run()
```

评估代码示例:所有model_type都需要传入model_path,['BERT',"ERNIE"]为保存模型所在目录，其他为模型文件

```py
from unlp import STextClassification
model = STextClassification(model_path='bert-base-chinese', model_type='BERT', mode='evaluate', datadir='./data/THUCNews', 
**kwargs)
res = model.run()
```

预测代码示例:所有model_type都需要传入model_path,['BERT',"ERNIE"]为保存模型所在目录，其他为模型文件  
**这时传入datadir的目的主要是为了加载datadir下的vocab文件，不会对数据进行加载**

```py
from unlp import STextClassification
model = STextClassification(model_path='bert-base-chinese', model_type='BERT', mode='predict', datadir='./data/THUCNews')
res = model.run(text=['艺龙网并购两家旅游网站',"封基上周溃退 未有明显估值优势","中华女子学院：本科层次仅1专业招男生"])
```

**如果需要对模型其他参数进行调节，可以参考相应模型文件中的Config类**  


### 2. 序列标注 
**命名实体识别** 
**通过model_path和model_type来制定模型**  
  model_path可以为相应的model名称:支持['bert-base-chinese', "chinese-bert-wwm-ext", "ernie-1.0", "albert-base-chinese"]或者是模型路径，如果为''，则会自动下载bert-base-chinese权重；  
  model_type目前支持['bert-base-chinese', "chinese-bert-wwm-ext", "ernie-1.0", "albert-base-chinese"]  
  mode为模型的三种模式：['train', "evaluate", "predict"]，分别对应于训练，评估和预测。
  data_dir为模型的输入数据，格式可以通过这个命令查看：  
```py
from unlp import NerDataFormat
```
**kwargs：额外需要传入的参数**  
如果是预测predict, run的参数需要传入text=List[str]这样的格式；   
如果是训练train,可以设置resume为True (bool类型）控制是否继续训练，其他预测predict和评估evaluate阶段可以不传入这个参数  

```py
from unlp import SEntityRecognition
model = SEntityRecognition(model_path, model_type, mode, datadir, **kwargs)
res = model.run()  # 实现模型的训练，评估和预测
```
finetune训练代码示例:需要传入model_path（为预训练模型所在目录或者通过字符串指定下载）  

```py
from unlp import SEntityRecognition
model = SEntityRecognition(model_path='bert-base-chinese', model_type='bert-base-chinese', mode='train', datadir='./data/THUCNews', 
**kwargs)
res = model.run()
```

resume训练代码示例:需要传入训练后的model_path,为保存模型所在目录  

```py
from unlp import SEntityRecognition
model = SEntityRecognition(model_path='./data/THUCNews/saved_dict/bert-base-chinese', model_type='bert-base-chinese', mode='train', datadir='./data/THUCNews', 
**{"resume":True})
res = model.run()
```

评估代码示例:所有model_type都需要传入model_path,为保存模型所在目录  

```py
from unlp import SEntityRecognition
model = STextClassification(model_path='./data/THUCNews/saved_dict/bert-base-chinese', model_type='bert-base-chinese', mode='evaluate', datadir='./data/THUCNews', 
**kwargs)
res = model.run()
```

预测代码示例:所有model_type都需要传入model_path,为保存模型所在目录    
**这时传入datadir的目的主要是为了加载datadir下的vocab文件，不会对数据进行加载**

```py
from unlp import SEntityRecognition
model = SEntityRecognition(model_path='./data/THUCNews/saved_dict/bert-base-chinese', model_type='bert-base-chinese', mode='predict', datadir='./data/THUCNews')
res = model.run(text=['艺龙网并购两家旅游网站',"封基上周溃退 未有明显估值优势","中华女子学院：本科层次仅1专业招男生"])
```

**如果需要对模型其他参数进行调节，可以tools下的config文件**


### 3. 文本生成  
**文章摘要生成** 
**通过model_path和model_type来制定模型**  
  model_path训练好的模型路径；  
  model_type目前支持['point-net"]  
  mode为模型的三种模式：['train', "evaluate", "predict"]，分别对应于训练，评估和预测。
  data_dir为模型的输入数据，格式可以通过这个命令查看：  
```py
from unlp import SummarizationDataFormat
```
**kwargs：额外需要传入的参数**  
如果是预测predict, run的参数需要传入text=List[str]这样的格式；   
如果是训练train,可以设置resume为True (bool类型）控制是否继续训练，其他预测predict和评估evaluate阶段可以不传入这个参数  

```py
from unlp import STextSummarization
model = STextSummarization(model_path, model_type, mode, datadir, **kwargs)
res = model.run()  # 实现模型的训练，评估和预测
```

训练代码示例:如果模型为空，则从头开始训练，如果继续训练resume需要传入训练后的model_path,为模型的路径  

```py
from unlp import STextSummarization
model = STextSummarization(model_path='./data/weibo/saved_dict/point-net/point-net.pt', model_type='point-net', mode='train', datadir='./data/weibo')
res = model.run()
```

评估代码示例:所有model_type都需要传入model_path,为保存模型所在目录,结果默认返回损失 

```py
from unlp import STextSummarization
model = STextSummarization(model_path='./data/weibo/saved_dict/point-net/point-net.pt', model_type='point-net', mode='evaluate', datadir='./data/weibo', 
**kwargs)
res = model.run()
```

评估代码示例:所有model_type都需要传入model_path,为保存模型所在目录，如果要进行rouge评估 

```py
from unlp import STextSummarization
model = STextSummarization(model_path='./data/weibo/saved_dict/point-net/point-net.pt', model_type='point-net', mode='evaluate', datadir='./data/weibo', 
**{"rouge":True, "refs":List[str], "preds":List[str]})
res = model.run()
```


预测代码示例:所有model_type都需要传入model_path,为保存模型所在目录    
**这时传入datadir的目的主要是为了加载datadir下的vocab文件，不会对数据进行加载**

```py
from unlp import STextSummarization
model = STextSummarization(model_path='./data/weibo/saved_dict/point-net/point-net.pt', model_type='point-net', mode='predict', datadir='./data/weibo')
res = model.run(text=["艺龙网并购两家旅游网站,封基上周溃退 未有明显估值优势,中华女子学院：本科层次仅1专业招男生"])
```

**如果需要对模型其他参数进行调节，可以gutils下的config文件**  

### 4. 文本对相关  
 
