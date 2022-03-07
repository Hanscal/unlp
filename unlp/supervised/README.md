# 监督学习

# 概述
    - 监督学习中有分类任务，命名实体识别，文本生成等方法；


# 文本分类
    - 1.一般模型相关：DPCNN，FastText，TextCNN，TextRNN，TextRCNN，TextRNN_Att。
    - 2.预训练语言模型相关：BERT和ERNIE

# 序列标注
    - 1.命名实体识别：通过Bert-BiLstm-CRF实现。

# 文本生成
    - 1.文本摘要：Pointer-Generator—Network
    - 2.对话生成：DialogGPT

# 文本对相关


# 使用方法
### 1. 文本分类  
    - **通过model_path和model_type来制定模型**  
    -   model_path可以为相应的model名称:支持['bert-base-chinese','ernie-1.0']或者是模型路径，如果为''，则会自动下载bert-base-chinese权重；  
    -   model_type目前支持['DPCNN', "FastText", "TextCNN", "TextRCNN", "TextRNN", "TextRNN_Att", "BERT", "ERNIE"]  
    -   mode为模型的三种模式：['train', "evaluate", "predict"]，分别对应于训练，评估和预测。
    -   datadir为模型的输入数据，格式可以通过这个命令查看：  
    
```py
from unlp import ClassificationDataFormat
```
    - **kwargs：额外需要传入的参数**  
    - 如果是预测predict, run的参数需要传入text=List[str]这样的格式；   
    - 如果是训练train,可以设置resume为True (bool类型）控制是否继续训练，其他预测predict和评估evaluate阶段可以不传入这个参数  

```py
from unlp import STextClassification
model = STextClassification(model_path, model_type, mode, datadir, **kwargs)
res = model.run()  # 实现模型的训练，评估和预测
```
    - finetune训练代码示例:['BERT',"ERNIE"]需要传入model_path（为预训练模型所在目录或者通过字符串指定下载）, 其他model_type不需要传入，model_type=''

```py
from unlp import STextClassification
model = STextClassification(model_path='bert-base-chinese', model_type='BERT', mode='train', datadir='./data/THUCNews', 
**kwargs)
res = model.run()
```

    - resume训练代码示例:所有model_type都需要传入model_path,['BERT',"ERNIE"]为保存模型所在目录，其他为模型文件

```py
from unlp import STextClassification
model = STextClassification(model_path='bert-base-chinese', model_type='BERT', mode='train', datadir='./data/THUCNews', 
**{"resume":True})
res = model.run()
```

    - 评估代码示例:所有model_type都需要传入model_path,['BERT',"ERNIE"]为保存模型所在目录，其他为模型文件

```py
from unlp import STextClassification
model = STextClassification(model_path='bert-base-chinese', model_type='BERT', mode='evaluate', datadir='./data/THUCNews', 
**kwargs)
res = model.run()
```

    - 预测代码示例:所有model_type都需要传入model_path,['BERT',"ERNIE"]为保存模型所在目录，其他为模型文件  
    - **这时传入datadir的目的主要是为了加载datadir下的vocab文件，不会对数据进行加载**

```py
from unlp import STextClassification
model = STextClassification(model_path='bert-base-chinese', model_type='BERT', mode='predict', datadir='./data/THUCNews')
res = model.run(text=['艺龙网并购两家旅游网站',"封基上周溃退 未有明显估值优势","中华女子学院：本科层次仅1专业招男生"])
```

    - **如果需要对模型其他参数进行调节，可以参考相应模型文件中的Config类**  


### 2. 序列标注 
**命名实体识别**   

    - **通过model_path和model_type来制定模型**  
    -   model_path可以为相应的model名称:支持['bert-base-chinese', "chinese-bert-wwm-ext", "ernie-1.0", "albert-base-chinese"]或者是模型路径，如果为''，则会自动下载bert-base-chinese权重；  
    -   model_type目前支持['bert-base-chinese', "chinese-bert-wwm-ext", "ernie-1.0", "albert-base-chinese"]  
    -   mode为模型的三种模式：['train', "evaluate", "predict"]，分别对应于训练，评估和预测。
    -   data_dir为模型的输入数据，格式可以通过这个命令查看： 
     
```py
from unlp import NerDataFormat
```
    - **kwargs：额外需要传入的参数**  
    - 如果是预测predict, run的参数需要传入text=List[str]这样的格式；   
    - 如果是训练train,可以设置resume为True (bool类型）控制是否继续训练，其他预测predict和评估evaluate阶段可以不传入这个参数  

```py
from unlp import SEntityRecognition
model = SEntityRecognition(model_path, model_type, mode, datadir, **kwargs)
res = model.run()  # 实现模型的训练，评估和预测
```

    - finetune训练代码示例:需要传入model_path（为预训练模型所在目录或者通过字符串指定下载）  

```py
from unlp import SEntityRecognition
model = SEntityRecognition(model_path='bert-base-chinese', model_type='bert-base-chinese', mode='train', datadir='./data/THUCNews', 
**kwargs)
res = model.run()
```

    - resume训练代码示例:需要传入训练后的model_path,为保存模型所在目录  

```py
from unlp import SEntityRecognition
model = SEntityRecognition(model_path='./data/THUCNews/saved_dict/bert-base-chinese', model_type='bert-base-chinese', mode='train', datadir='./data/THUCNews', 
**{"resume":True})
res = model.run()
```

    - 评估代码示例:所有model_type都需要传入model_path,为保存模型所在目录  

```py
from unlp import SEntityRecognition
model = STextClassification(model_path='./data/THUCNews/saved_dict/bert-base-chinese', model_type='bert-base-chinese', mode='evaluate', datadir='./data/THUCNews', 
**kwargs)
res = model.run()
```

    - 预测代码示例:所有model_type都需要传入model_path,为保存模型所在目录    
    - **这时传入datadir的目的主要是为了加载datadir下的vocab文件，不会对数据进行加载**

```py
from unlp import SEntityRecognition
model = SEntityRecognition(model_path='./data/THUCNews/saved_dict/bert-base-chinese', model_type='bert-base-chinese', mode='predict', datadir='./data/THUCNews')
res = model.run(text=['艺龙网并购两家旅游网站',"封基上周溃退 未有明显估值优势","中华女子学院：本科层次仅1专业招男生"])
```

    - **如果需要对模型其他参数进行调节，可以tools下的config文件**


### 3. 文本生成  
**文章摘要生成**   

    - **通过model_path和model_type来制定模型**  
    -   model_path训练好的模型路径；  
    -   model_type目前支持['point-net"]  
    -   mode为模型的三种模式：['train', "evaluate", "predict"]，分别对应于训练，评估和预测。
    -   data_dir为模型的输入数据，格式可以通过这个命令查看： 
     
```py
from unlp import SummarizationDataFormat
```
    - **kwargs：额外需要传入的参数**  
    - 如果是预测predict, run的参数需要传入text=List[str]这样的格式；   
    - 如果是训练train,可以设置resume为True (bool类型）控制是否继续训练，其他预测predict和评估evaluate阶段可以不传入这个参数  

```py
from unlp import STextSummarization
model = STextSummarization(model_path, model_type, mode, datadir, **kwargs)
res = model.run()  # 实现模型的训练，评估和预测
```

    - 训练代码示例:如果模型为空，则从头开始训练，如果继续训练resume需要传入训练后的model_path,为模型的路径  

```py
from unlp import STextSummarization
model = STextSummarization(model_path='./data/weibo/saved_dict/point-net/point-net.pt', model_type='point-net', mode='train', datadir='./data/weibo')
res = model.run()
```

    - 评估代码示例:所有model_type都需要传入model_path,为保存模型所在目录,结果默认返回损失 

```py
from unlp import STextSummarization
model = STextSummarization(model_path='./data/weibo/saved_dict/point-net/point-net.pt', model_type='point-net', mode='evaluate', datadir='./data/weibo', 
**kwargs)
res = model.run()
```

    - 评估代码示例:所有model_type都需要传入model_path,为保存模型所在目录，如果要进行rouge评估 

```py
from unlp import STextSummarization
model = STextSummarization(model_path='./data/weibo/saved_dict/point-net/point-net.pt', model_type='point-net', mode='evaluate', datadir='./data/weibo', 
**{"rouge":True, "refs":List[str], "preds":List[str]})
res = model.run()
```


    - 预测代码示例:所有model_type都需要传入model_path,为保存模型所在目录    
    - **这时传入datadir的目的主要是为了加载datadir下的vocab文件，不会对数据进行加载**

```py
from unlp import STextSummarization
model = STextSummarization(model_path='./data/weibo/saved_dict/point-net/point-net.pt', model_type='point-net', mode='predict', datadir='./data/weibo')
res = model.run(text=["艺龙网并购两家旅游网站,封基上周溃退 未有明显估值优势,中华女子学院：本科层次仅1专业招男生"])
```

    - **如果需要对模型其他参数进行调节，可以gutils下的config文件** 

**对话生成dialog-gpt**   

    - **通过model_path和model_type来制定模型**  
    -   model_path训练好的模型路径；  
    -   model_type目前支持['dialog-gpt"]  
    -   mode为模型的三种模式：['train', "evaluate", "predict"]，分别对应于训练，评估和预测。
    -   data_dir为模型的输入数据，格式可以通过这个命令查看：  
```py
from unlp import DialogueDataFormat
```
    - **kwargs：额外需要传入的参数**  
    - 如果是预测predict, run的参数需要传入text=List[str]这样的格式；   
    - 如果是训练train,可以设置resume为True (bool类型）控制是否继续训练，其他预测predict和评估evaluate阶段可以不传入这个参数  

```py
from unlp import SDialogueGeneration
model = SDialogueGeneration(model_path, model_type, mode, datadir, **kwargs)
res = model.run()  # 实现模型的训练，评估和预测
```

    - 训练代码示例:如果模型为空，则从头开始训练，如果继续训练resume需要传入训练后的model_path,为模型的路径  

```py
from unlp import SDialogueGeneration
model = SDialogueGeneration(model_path='./data/weibo/saved_dict/point-net/point-net.pt', model_type='point-net', mode='train', datadir='./data/weibo')
res = model.run()
```

    - 评估代码示例:所有model_type都需要传入model_path,为保存模型所在目录,结果默认返回损失 

```py
from unlp import SDialogueGeneration
model = SDialogueGeneration(model_path='./data/weibo/saved_dict/point-net/point-net.pt', model_type='point-net', mode='evaluate', datadir='./data/weibo', 
**kwargs)
res = model.run()
```

    - 评估代码示例:所有model_type都需要传入model_path,为保存模型所在目录，如果要进行rouge评估 

```py
from unlp import SDialogueGeneration
model = SDialogueGeneration(model_path='./data/weibo/saved_dict/point-net/point-net.pt', model_type='point-net', mode='evaluate', datadir='./data/weibo', 
**{"rouge":True, "refs":List[str], "preds":List[str]})
res = model.run()
```


    - 预测代码示例:所有model_type都需要传入model_path,为保存模型所在目录    
    - **这时传入datadir的目的主要是为了加载datadir下的vocab文件，不会对数据进行加载**

```py
from unlp import SDialogueGeneration
model = SDialogueGeneration(model_path='./data/weibo/saved_dict/point-net/point-net.pt', model_type='point-net', mode='predict', datadir='./data/weibo')
res = model.run(text=["艺龙网并购两家旅游网站,封基上周溃退 未有明显估值优势,中华女子学院：本科层次仅1专业招男生"])
```

    - **如果需要对模型其他参数进行调节，可以gutils下的config文件** 


### 4. 文本对相关  
   
# 相关论文  
## 文本分类
[1] Convolutional Neural Networks for Sentence Classification  
[2] Recurrent Neural Network for Text Classification with Multi-Task Learning  
[3] Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification  
[4] Recurrent Convolutional Neural Networks for Text Classification  
[5] Bag of Tricks for Efficient Text Classification  
[6] Deep Pyramid Convolutional Neural Networks for Text Categorization  
[7] Attention Is All You Need  

## 文本生成
[1] Pointer-Generator—Network (Get To The Point: Summarization with Pointer-Generator Networks).  
[2] DialoGPT (DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation).
