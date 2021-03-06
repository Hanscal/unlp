<h1 align="center"><a href="https://github.com/hanscal/unlp" target="_blank">unlp</a></h1>

<div align="center">

[![PyPI version](https://badge.fury.io/py/unlp.svg)](https://badge.fury.io/py/unlp)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/unlp.svg)](https://pypi.python.org/pypi/unlp)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/Hanscal/unlp/pulls)
</div>

<p align="center">
  <a href="https://github.com/Hanscal/unlp/stargazers"><img alt="star" src="https://img.shields.io/github/stars/Hanscal/unlp.svg?label=Stars&style=social"/></a>
  <a href="https://github.com/Hanscal/unlp/network/members"><img alt="star" src="https://img.shields.io/github/forks/Hanscal/unlp.svg?label=Fork&style=social"/></a>
  <a href="https://github.com/Hanscal/unlp/watchers"><img alt="star" src="https://img.shields.io/github/watchers/Hanscal/unlp.svg?label=Watch&style=social"/></a>
  
</p>

<div align="center">
  
`gitHub`欢迎提[pull requests](https://github.com/Hanscal/unlp/pulls), 如果有`bug`或`新需求`，请反馈[issue](https://github.com/Hanscal/unlp/issues)
</div>

## unlp

unlp是一些经常需要用到的NLP算法包，有助于您学习和使用基于深度学习的文本处理。

## 安装
**python3.6+**

```py
pip install unlp 
pip install -r requirements.txt
```

## unlp模块
----
1. 根据nlp的任务来构建这个包，主要有无监督学习、有监督学习以及文本增强的一些常有方法；
2. 无监督学习中有关键词抽取，向量嵌入和相似度计算；  
3. 监督学习中有分类任务，命名实体识别，文本生成等；  
4. 文本增强常用方法，比如回译，同义词替换等。

## unlp使用
1. 具体参见[示例目录](https://github.com/Hanscal/unlp/tree/master/examples)  
2. 项目中有的任务需要自动下载预训练模型，在百度网盘提供部分已经下载了的模型：  
[预训练模型](https://pan.baidu.com/s/1gXv14q-uzQb9urhveQKzeg)  密码: 6lk1

## 无监督学习方法
[详细使用说明](https://github.com/Hanscal/unlp/blob/master/unlp/unsupervised/README.md)
1. 关键词抽取；  
2. 向量嵌入；  
3. 相似度计算；  
4. 语义搜索；  
...

## 监督学习方法
[详细使用说明](https://github.com/Hanscal/unlp/blob/master/unlp/supervised/README.md)
1. 文本分类； 
2. 命名实体识别；  
3. 文本摘要；  
4. 对话生成；  
...

## 文本数据增强  
[详细使用说明](https://github.com/Hanscal/unlp/blob/master/unlp/augment/README.md)
1. 回译；  
2. EDA(同义词替换、插入、交换和删除)；   
...


 
