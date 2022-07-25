```python
import os
from unlp import DataAugment
```

    /home/ubuntu/anaconda3/envs/unlp/lib/python3.6/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


    
     Synonyms: v3.18.0, Project home: https://github.com/chatopera/Synonyms/
    
     Project Sponsored by Chatopera
    
      deliver your chatbots with Chatopera Cloud Services --> https://bot.chatopera.com
    
    >> Synonyms load wordseg dict [/home/ubuntu/anaconda3/envs/unlp/lib/python3.6/site-packages/synonyms/data/vocab.txt] ... 
    >> Synonyms on loading stopwords [/home/ubuntu/anaconda3/envs/unlp/lib/python3.6/site-packages/synonyms/data/stopwords.txt] ...
    >> Synonyms on loading vectors [/home/ubuntu/anaconda3/envs/unlp/lib/python3.6/site-packages/synonyms/data/words.vector.gz] ...


# 语言模型生成数据

## BERT


```python
model_path = '/data/lss/deepenv/deepenv-data/unlp_debug/transformer/bert-base-chinese'
model = DataAugment(model_path=model_path, mode='BERT', num_aug=5)
model.run(text='哪款可以保留矿物质')
```

    Some weights of the model checkpoint at /data/lss/deepenv/deepenv-data/unlp_debug/transformer/bert-base-chinese were not used when initializing BertForMaskedLM: ['cls.seq_relationship.bias', 'cls.seq_relationship.weight']
    - This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


    cost 0.7421634197235107





    ['哪款可以补充矿物质',
     '哪款可以保留矿物质',
     '哪款可以保留一年？',
     '哪款可以不保留矿物质',
     '哪款可以只保留矿物质',
     '哪款可以？保留矿物质']



## BART


```python
model_path = '/data/lss/deepenv/deepenv-data/unlp_debug/transformer/bart-base-chinese'
model = DataAugment(model_path=model_path, mode='BART', num_aug=5)
model.run(text='哪款可以保留矿物质')
```

    The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
    The tokenizer class you load from this checkpoint is 'BartTokenizer'. 
    The class this function is called from is 'BertTokenizer'.


    cost 1.1044518947601318





    ['还可以保留矿物质',
     '哪款可以补充矿物质',
     '哪款可以保留下来',
     '哪款可以保留的矿物质',
     '没有哪款可以保留矿物质',
     '哪款饮料可以保留矿物质']



# EDA


```python
model_path = '/data/lss/deepenv/deepenv-data/unlp_debug/transformer/word2vec/light_Tencent_AILab_ChineseEmbedding.bin'
model = DataAugment(model_path=model_path, mode='EDA', num_aug=5)
model.run(text='哪款可以保留矿物质')
```

    loading word2vec model from /data/lss/deepenv/deepenv-data/unlp_debug/transformer/word2vec/light_Tencent_AILab_ChineseEmbedding.bin
    cost 0.000919342041015625





    ['哪款可以保留矿物质', '此款可以保留碳水化合物', '哪款保留可以矿物质', '哪款可以保留矿物质', '哪款可以保留矿物质']



# 回译


```python
model = DataAugment(mode='Translate')
model.run(text='哪款可以保留矿物质')
```

    cost 1.9441919326782227





    ['哪一种可以保留矿物质']




```python

```
