```python
import torch 
flag = torch.cuda.is_available() 
print(flag) 

ngpu= 1 
# Decide which device we want to run on
device = torch.device("cuda:3" if (torch.cuda.is_available() and ngpu > 0) else "cpu") 
print(device) 
print(torch.cuda.get_device_name(0)) 
print(torch.rand(3,3).cuda()) 
```

# Classification

- **通过model_path和model_type来制定模型**  
-   model_path可以为相应的model名称:支持['bert-base-chinese','ernie-1.0']或者是模型路径，如果为''，则会自动下载bert-base-chinese权重；  
-   model_type目前支持['DPCNN', "FastText", "TextCNN", "TextRCNN", "TextRNN", "TextRNN_Att", "BERT", "ERNIE"]  
-   mode为模型的三种模式：['train', "evaluate", "predict"]，分别对应于训练，评估和预测。


```python
from unlp import STextClassification
model = STextClassification('../transformer/bert-base-chinese/', "BERT", 'train', './unlp/supervised/classification/data/THUCNews')
res = model.run()  # 实现模型的训练，评估和预测
```

## BERT


```python
from unlp import STextClassification
# train
model = STextClassification(model_path='../transformer/bert-base-chinese/', model_type='BERT', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()
#train resume 
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/BERT', model_type='BERT', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()
# evaluate
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/BERT/', model_type='BERT', mode='evaluate', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()


# predict
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/BERT/', model_type='BERT', mode='predict', datadir='./unlp/supervised/classification/data/THUCNews', )
model.run(text=['艺龙网并购两家旅游网站',"封基上周溃退 未有明显估值优势","中华女子学院：本科层次仅1专业招男生"])
```


```python
from unlp import STextClassification

#train resume
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/bert', model_type='BERT', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()
```

## ERNIE


```python
from unlp import STextClassification
#train
model = STextClassification(model_path='../transformer/ernie-1.0/', model_type='ERNIE', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()
#train resume
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/ERNIE', model_type='ERNIE', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()
#evaluate
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/ERNIE/', model_type='ERNIE', mode='evaluate', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()


#predict
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/ERNIE/', model_type='ERNIE', mode='predict', datadir='./unlp/supervised/classification/data/THUCNews', )
model.run(text=['艺龙网并购两家旅游网站',"封基上周溃退 未有明显估值优势","中华女子学院：本科层次仅1专业招男生"])
```


```python

```

## DPCNN


```python
from unlp import STextClassification
#train
model = STextClassification(model_path='', model_type='DPCNN', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()
#train resume
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/DPCNN.ckpt', model_type='DPCNN', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()
#evaluate
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/DPCNN.ckpt', model_type='DPCNN', mode='evaluate', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()


#predict
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/DPCNN.ckpt', model_type='DPCNN', mode='predict', datadir='./unlp/supervised/classification/data/THUCNews', )
model.run(text=['艺龙网并购两家旅游网站',"封基上周溃退 未有明显估值优势","中华女子学院：本科层次仅1专业招男生"])
```


```python

```

## FastText


```python
from unlp import STextClassification
# train
model = STextClassification(model_path='', model_type='FastText', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()
#train resume
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/FastText.ckpt', model_type='FastText', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()
# evaluate
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/FastText.ckpt', model_type='FastText', mode='evaluate', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()


#predict
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/FastText.ckpt', model_type='FastText', mode='predict', datadir='./unlp/supervised/classification/data/THUCNews', )
model.run(text=['艺龙网并购两家旅游网站',"封基上周溃退 未有明显估值优势","中华女子学院：本科层次仅1专业招男生"])
```

## TextCNN


```python
from unlp import STextClassification
#train
model = STextClassification(model_path='', model_type='TextCNN', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()
#train resume
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/TextCNN.ckpt', model_type='TextCNN', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()
#evaluate
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/TextCNN.ckpt', model_type='TextCNN', mode='evaluate', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()


# # #predict
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/TextCNN.ckpt', model_type='TextCNN', mode='predict', datadir='./unlp/supervised/classification/data/THUCNews', )
model.run(text=['艺龙网并购两家旅游网站',"封基上周溃退 未有明显估值优势","中华女子学院：本科层次仅1专业招男生"])
```

## TextRCNN


```python
from unlp import STextClassification
#train
model = STextClassification(model_path='', model_type='TextRCNN', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()
#train resume

model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/TextRCNN.ckpt', model_type='TextRCNN', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()

#evaluate
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/TextRCNN.ckpt', model_type='TextRCNN', mode='evaluate', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()


# #predict
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/TextRCNN.ckpt', model_type='TextRCNN', mode='predict', datadir='./unlp/supervised/classification/data/THUCNews', )
model.run(text=['艺龙网并购两家旅游网站',"封基上周溃退 未有明显估值优势","中华女子学院：本科层次仅1专业招男生"])
```

## TextRNN


```python
from unlp import STextClassification
#train
model = STextClassification(model_path='', model_type='TextRNN', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()
#train resume

model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/TextRNN.ckpt', model_type='TextRNN', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()
#evaluate
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/TextRNN.ckpt', model_type='TextRNN', mode='evaluate', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()


# #predict
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/TextRNN.ckpt', model_type='TextRNN', mode='predict', datadir='./unlp/supervised/classification/data/THUCNews', )
model.run(text=['艺龙网并购两家旅游网站',"封基上周溃退 未有明显估值优势","中华女子学院：本科层次仅1专业招男生"])
```

## TextRNN_Att


```python
from unlp import STextClassification
#train
model = STextClassification(model_path='', model_type='TextRNN_Att', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()
#train resume

model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/TextRNN_Att.ckpt', model_type='TextRNN_Att', mode='train', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()
#evaluate
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/TextRNN_Att.ckpt', model_type='TextRNN_Att', mode='evaluate', datadir='./unlp/supervised/classification/data/THUCNews', )
res = model.run()


# #predict
model = STextClassification(model_path='./unlp/supervised/classification/data/THUCNews/saved_dict/TextRNN_Att.ckpt', model_type='TextRNN_Att', mode='predict', datadir='./unlp/supervised/classification/data/THUCNews', )
model.run(text=['艺龙网并购两家旅游网站',"封基上周溃退 未有明显估值优势","中华女子学院：本科层次仅1专业招男生"])
```
