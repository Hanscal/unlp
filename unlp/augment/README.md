# 文本增强

# 概述
    - 相较于图像数据增强，文本数据增强，现在还是有很多问题的；
    - 往更严格的角度看，文本数据增强更像是同义句生成，但又不完全是，它是一个更大范围的概念；
    - 很多时候，需要文本数据增强，一个是常常遇到的数据不足，另一个就是数据不均衡。
    - 文本数据增强的有效方法:
    - 一个是回译（翻译两次，例如中文到英文，然后英文再到中文），比较适用短文本；
    - 一个是EDA（easy data augmentation：同义词替换、插入、交换和删除），比较适用长文本；

# 回译
    - 1.在线翻译工具（中文->[英、法、德、俄、西班牙、葡萄牙、日、韩、荷兰、阿拉伯]等语言）
       **百度翻译(baidu)**，百度翻译不用说，国内支持翻译语言最多的了(28种互译)，而且最大方了，注册账户后每月有200万字符的流量，大约是2M吧，超出则49元人民币/百万字符
       - 谷歌翻译(google)，谷歌翻译不用说，应该是挺好的，语言支持最多，不过需要翻墙注册账户
    - 2.离线翻译工具
       - 1.自己写，收集些语料，seq2seq,nmt,transformer

# 同义词EDA
    - 1.eda(其实就是同义词替换、插入、交换和删除)
        - 使用了synonyms工具查找同义词
    - 2.word2vec、词典同义词替换
        - 使用gensim的词向量，找出某个词最相似的词作为同义词。

# LM生成：
    - 1. 原始query的token，不过会做一些随机mask，来预测mask掉的词语。
    - 2. 分类的类别标签，保证语义不变性。

分类的类别标签，保证语义不变性。
   
# 使用方法
**对于短文本，回译效果比较好。对于长文本，同义词EDA效果比较好。**  
 
### 回译  

### 同义词替换  

### LM生成方法  


**[其他方法]**  
* nlpcda(https://github.com/425776024/nlpcda)  
* TextAttack(https://github.com/QData/TextAttack)   
* nlpaug(https://github.com/makcedward/nlpaug)

 
# 相关论文  
[1] 《Easy data augmentation techniques for boosting performance on text classification tasks》
[2] 《Contextual Augmentation:Data Augmentation by Words with Paradigmatic Relations》
[3] 《Conditional BERT Contextual Augmentation》
[4] 《TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP》