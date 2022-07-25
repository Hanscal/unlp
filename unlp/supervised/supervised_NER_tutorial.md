# NER

- **通过model_path和model_type来制定模型**  
-   model_path可以为相应的model名称:支持['bert-base-chinese', "chinese-bert-wwm-ext", "ernie-1.0", "albert-base-chinese"]或者是模型路径，如果为''，则会自动下载bert-base-chinese权重；  
-   model_type目前支持['bert-base-chinese', "chinese-bert-wwm-ext", "ernie-1.0", "albert-base-chinese"]  
-   mode为模型的三种模式：['train', "evaluate", "predict"]，分别对应于训练，评估和预测。

## bert-base-chinese


```python
from unlp import SEntityRecognition
# train
# model = SEntityRecognition(model_path='/data/lss/deepenv/deepenv-data/unlp_debug/transformer/bert-base-chinese/', model_type='bert-base-chinese', mode='train', datadir='/data/lss/deepenv/deepenv-data/unlp_debug/unlp/supervised/ner/data/cluner/', )
# res = model.run()

# train resume 注：需将训练好的模型拷贝到其他路径，不可为{datadir}/saved_dict/bert-base-chinese路径
# model = SEntityRecognition(model_path='./unlp/supervised/ner/data/cluner/saved_dict/bert', model_type='bert-base-chinese', mode='train', datadir='./unlp/supervised/ner/data/cluner/', )
# res = model.run()

# evaluate
# model = SEntityRecognition(model_path='./unlp/supervised/ner/data/cluner/saved_dict/bert-base-chinese', model_type='bert-base-chinese', mode='evaluate', datadir='./unlp/supervised/ner/data/cluner/', )
# res = model.run()
# print(res)

# predict
# model = SEntityRecognition(model_path='./unlp/supervised/ner/data/cluner/saved_dict/bert-base-chinese', model_type='bert-base-chinese', mode='predict', datadir='./unlp/supervised/ner/data/cluner/', )
# res = model.run(text = ["是否有一种更加合理和安全的方式能够保护消费者的用卡安全，值得信用卡发行者探讨。","除了资金支持之外，工行还为创业大学生提供包括存款、资金结算、电子银行、银行卡等一站式金融服务，"])
# print(res)
```

## chinese-bert-wwm-ext


```python
from unlp import SEntityRecognition
# model = SEntityRecognition(model_path='/data/lss/deepenv/deepenv-data/unlp_debug/transformer/chinese-bert-wwm-ext', model_type='chinese-bert-wwm-ext', mode='train', datadir='/data/lss/deepenv/deepenv-data/unlp_debug/unlp/supervised/ner/data/cluner/', )
# res = model.run()

# train resume 注：需将训练好的模型拷贝到其他路径，不可为{datadir}/saved_dict/chinese-bert-wwm-ext路径
# model = SEntityRecognition(model_path='./unlp/supervised/ner/data/cluner/saved_dict/chinese-bert-wwm', model_type='chinese-bert-wwm-ext', mode='train', datadir='./unlp/supervised/ner/data/cluner/', )
# res = model.run()

# evaluate
# model = SEntityRecognition(model_path='./unlp/supervised/ner/data/cluner/saved_dict/chinese-bert-wwm-ext', model_type='chinese-bert-wwm-ext', mode='evaluate', datadir='./unlp/supervised/ner/data/cluner/', )
# res = model.run()
# print(res)

# predict
# model = SEntityRecognition(model_path='./unlp/supervised/ner/data/cluner/saved_dict/chinese-bert-wwm-ext', model_type='chinese-bert-wwm-ext', mode='predict', datadir='./unlp/supervised/ner/data/cluner/', )
# res = model.run(text = ["是否有一种更加合理和安全的方式能够保护消费者的用卡安全，值得信用卡发行者探讨。","除了资金支持之外，工行还为创业大学生提供包括存款、资金结算、电子银行、银行卡等一站式金融服务，"])
# print(res)
```

## albert-base-chinese


```python
from unlp import SEntityRecognition
model = SEntityRecognition(model_path='/data/lss/deepenv/deepenv-data/unlp_debug/transformer/albert-base-chinese', model_type='albert-base-chinese', mode='train', datadir='/data/lss/deepenv/deepenv-data/unlp_debug/unlp/supervised/ner/data/cluner/', )
res = model.run()

# train resume 注：需将训练好的模型拷贝到其他路径，不可为{datadir}/saved_dict/albert-base-chinese路径
# model = SEntityRecognition(model_path='./unlp/supervised/ner/data/cluner/saved_dict/albert, model_type='albert-base-chinese', mode='train', datadir='./unlp/supervised/ner/data/cluner/', )
# res = model.run()

# evaluate
# model = SEntityRecognition(model_path='./unlp/supervised/ner/data/cluner/saved_dict/albert-base-chinese', model_type='albert-base-chinese', mode='evaluate', datadir='./unlp/supervised/ner/data/cluner/', )
# res = model.run()
# print(res)

# predict
# model = SEntityRecognition(model_path='./unlp/supervised/ner/data/cluner/saved_dict/albert-base-chinese', model_type='albert-base-chinese', mode='predict', datadir='./unlp/supervised/ner/data/cluner/', )
# res = model.run(text = ["是否有一种更加合理和安全的方式能够保护消费者的用卡安全，值得信用卡发行者探讨。","除了资金支持之外，工行还为创业大学生提供包括存款、资金结算、电子银行、银行卡等一站式金融服务，"])
# print(res)

```

## ernie-1.0


```python
from unlp import SEntityRecognition
# model = SEntityRecognition(model_path='/data/lss/deepenv/deepenv-data/unlp_debug/transformer/ernie-1.0', model_type='ernie-1.0', mode='train', datadir='/data/lss/deepenv/deepenv-data/unlp_debug/unlp/supervised/ner/data/cluner/', )
# res = model.run()

# train resume 注：需将训练好的模型拷贝到其他路径，不可为{datadir}/saved_dict/ernie-1.0路径
# model = SEntityRecognition(model_path='./unlp/supervised/ner/data/cluner/saved_dict/ernie, model_type='ernie-1.0', mode='train', datadir='./unlp/supervised/ner/data/cluner/', )
# res = model.run()

# evaluate
# model = SEntityRecognition(model_path='./unlp/supervised/ner/data/cluner/saved_dict/ernie-1.0', model_type='ernie-1.0', mode='evaluate', datadir='./unlp/supervised/ner/data/cluner/', )
# res = model.run()
# print(res)

# predict
# model = SEntityRecognition(model_path='./unlp/supervised/ner/data/cluner/saved_dict/ernie-1.0', model_type='ernie-1.0', mode='predict', datadir='./unlp/supervised/ner/data/cluner/', )
# res = model.run(text = ["是否有一种更加合理和安全的方式能够保护消费者的用卡安全，值得信用卡发行者探讨。","除了资金支持之外，工行还为创业大学生提供包括存款、资金结算、电子银行、银行卡等一站式金融服务，"])
# print(res)
```


```python

```
