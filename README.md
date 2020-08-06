目前有一些自然语言处理工具，可以快速地完成NLP任务，如：
1. `NLTK`:处于领先的地位，提供了 WordNet 、分类、分词、标注、语法分析、语义推理等类库。
2. `Pattern`:包括词性标注工具，N元搜索，情感分析，WordNet，支持机器学习的向量空间模型，聚类，向量机。
3. `TextBlob`:提供了一些简单的api，例如词性标注、名词短语抽取、情感分析、分类、翻译等等。
4. `Gensim`:提供了对大型语料库的主题建模、文件索引、相似度检索的功能。
5. `PyNLPI`可以用来处理N元搜索，计算频率表和分布，建立语言模型。
6. `spaCy`:结合Python和Cython，它的自然语言处理能力达到了工业强度。
7. `Polyglot`支持对海量文本和多语言的处理。
8. `MontyLingua`:端到端的英文处理工具。适合用来进行信息检索和提取，问题处理，回答问题等任务。从英文文本中，它能提取出主动宾元组，形容词、名词和动词短语，人名、地名、事件，日期和时间，等语义信息。
9. `BLLIP Parser`:集成了产生成分分析和最大熵排序的统计自然语言工具。
10. `Quepy`:轻松地实现不同类型的自然语言和数据库查询语言的转化。
11. `HanNLP`:由一系列模型与算法组成的Java工具包，提供分词、词法分析、句法分析、语义理解等完备的功能。
12. `Jiagu`:以BiLSTM等模型为基础，使用大规模语料训练而成，提供中文分词、词性标注、命名实体识别、情感分析、知识图谱关系抽取、关键词抽取、文本摘要、新词发现、情感分析、文本聚类等常用功能。

这里整理一些知名的NLP开源项目，分别为：

#### 词义消歧
消歧任务主要分以下几种：
1. 分词的消歧
2. 多义词的具体词义
3. 词性的判断

对于词性的判断，可以看做一个词性标注的问题，通常需要考虑邻近上下文。如果是词义判决，可能会有相隔很远的词语来决定其词性。因此大部分的词性标注模型简单地使用当前上下文，而语义消歧模型通常使用规模广泛一些的上下文中的实词。

**消歧方法分为以下几种：**
- 有监督消歧
- 无监督消歧
- 基于词典的消歧

[WordMultiSenseDisambiguation](https://github.com/liuhuanyong/WordMultiSenseDisambiguation)
WordMultiSenseDisambiguation, chinese multi-wordsense disambiguation based on online bake knowledge base and semantic embedding similarity compute,基于百科知识库的中文词语多义项获取与特定句子词语语义消歧. 

#### 拼写纠错
[bert_chinese-master](https://github.com/JohanyCheung/bert_chinese/tree/master/corrector)
直接预训练的bert模型实现中文的文本纠错，可参照学习bert如何做纠错任务
[FASPell-master](https://github.com/iqiyi/FASPell/blob/master)
使用bert进行预训练+微调，再经过CSD过滤器得到最终结果。支持简体中文文本； 繁体中文文本； 人类论文； OCR结果等
[pycorrector-master](https://github.com/shibing624/pycorrector)
Pycorrector：当前主流的中文纠错框架，支持规则和端到端模型
[SoftMaskedBert-master](https://github.com/hiyoung123/SoftMaskedBert)
对论文Soft-Masked Bert 的复现，使用判别模型BiGRU+纠错模型BERT，实现端到端的纠错。
[YoungCorrector-master](https://github.com/hiyoung123/YoungCorrector)
基于Pycorrector改造，实现基于纯规则的纠错系统。整个系统框架比较详细。与Pycorrector准确度差不多，耗时短（归功于前向最大匹配替代了直接索引混淆词典）

#### 文本分类
**文本分类常用方法如下：**
1. 基于词典模板的规则分类
2. 基于过往日志匹配（适用于搜索引擎）
3. 基于传统机器学习模型(特征工程+算法，如NaiveBayes/SVM/LR/KNN......)
4. 基于深度学习模型:词向量+模型(FastText/TextCNN/TextRNN/TextRCNN/DPCNN/BERT/VDCNN)

这几种方式基本上是目前比较主流的方法，现在进行文本分类的难点主要是两点，一点是数据来源的匮乏，因为方法已经比较固定，基本都是有监督学习，需要很多的标记数据，现在我们常用的数据要么就是找专业标记团队去买，要么就是自己去爬。第二点是尽管是分类工作，但是文本分类种类很多，并且要求的准确性，拓展性都不是之前的分类可比的，这一点也是很困难的。

| 方法 | 描述 |
| :---- | :---- |
| 朴素贝叶斯 | 朴素贝叶斯法( naive Bayes)可算是最简单常用的一种生成式模型。朴素贝叶斯法基于贝叶斯定理将联合概率转化为条件概率，然后利用特征条件独立假设简化条件概率的计算。 |
| SVM | 找出一个决策边界，使得边界到正负样本的最小距离都最远。 |
| fastText  | fastText原理是把句子中所有的词进行lookup得到词向量之后，对向量进行平均（某种意义上可以理解为只有一个avgpooling特殊CNN），然后直接接softmax层预测label。在label比较多的时候，为了降低计算量，论文最后一层采用了层次softmax的方法，既根据label的频次建立哈夫曼树，每个label对应一个哈夫曼编码，每个哈夫曼树节点具有一个向量作为参数进行更新，预测的时候隐层输出与每个哈夫曼树节点向量做点乘，根据结果决定向左右哪个方向移动，最终落到某个label对应的节点上。 |
| TextCNN  | 首先，对句子做padding或者截断，保证句子长度为固定值s=7,单词embedding成d=5维度的向量，这样句子被表示为(s,d)(s,d)大小的矩阵（类比图像中的像素）。然后经过有filter_size=(2,3,4)的一维卷积层，每个filter_size有两个输出channel。第三层是一个1-maxpooling层，这样不同长度句子经过pooling层之后都能变成定长的表示了，最后接一层全连接的softmax层，输出每个类别的概率。 |
| TextRNN	 | 对于英文，都是基于词的。对于中文，首先要确定是基于字的还是基于词的。如果是基于词，要先对句子进行分词。之后，每个字/词对应RNN的一个时刻，隐层输出作为下一时刻的输入。最后时刻的隐层输出h_ThTcatch住整个句子的抽象特征，再接一个softmax进行分类。|
| TextRNN+Attention	| 在TextRNN的基础上加入了attention机制 | 
| TextRCNN	| 利用前向和后向RNN得到每个词的前向和后向上下文的表示，词的表示就变成词向量和前向后向上下文向量concat起来的形式，最后再接跟TextCNN相同卷积层，pooling层即可，唯一不同的是卷积层filter_size=1就可以了，不再需要更大filter_size获得更大视野，这里词的表示也可以只用双向RNN输出| 
| HAN	| HAN模型首先利用Bi-GRU捕捉单词级别的上下文信息。由于句子中的每个单词对于句子表示并不是同等的贡献，因此，作者引入注意力机制来提取对句子表示有重要意义的词汇，并将这些信息词汇的表征聚合起来形成句子向量。| 
| BERT| BERT的模型架构是一个多层的双向Transformer编码器(Transformer的原理及细节可以参考Attentionisallyouneed)。作者采用两套参数分别生成BERTBASE模型和BERTLARGE模型(细节描述可以参考原论文)，所有下游任务可以在这两套模型进行微调。| 
| VDCNN	| 目前NLP领域的模型，无论是机器翻译、文本分类、序列标注等问题大都使用浅层模型。VDCNN探究的是深层模型在文本分类任务中的有效性，最优性能网络达到了29层。| 
[Chinese-Text-Classification-Pytorch-master](https://github.com/649453932/Chinese-Text-Classification-Pytorch)
中文文本分类，TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention, DPCNN, Transformer, 基于pytorch，开箱即用。
[Naive-Bayes-Classifier-master](https://github.com/lining0806/Naive-Bayes-Classifier)
使用朴素贝叶斯进行文本分类：
[text-classification-cnn-rnn-master](https://github.com/gaussic/text-classification-cnn-rnn)
使用CNN以及RNN进行中文文本分类
[text-cnn-master](https://github.com/cjymz886/text-cnn)
使用CNN模型进行文本分类

#### 序列标注
序列标注包括了以下几个方面：
1. `中文分词`——人们提出了{B,M,E,S}这种最流行的标注集，{B,M,E,S}分为代表{Begin,Middle,End,Single}
2. `词性标注`——根据单词序列，标注出词性序列
3. `命名实体识别`——命名实体识别可以复用{B,M,E,S}标注集，但是还需要确定实体所属的类别

`序列标注方面`，传统的统计方法是HMM(隐马尔可夫)、MEMM(最大熵马尔可夫)、CRF(条件随机场)，基本就和命名实体识别类似了，而在深度学习引入后，形成了输入层、编码层、解码层的主要架构，同过预训练表征模型(如w2v)、深度学习结构(CNN、RNN等)以及输出层(CRF、softmax)等结构链接，完成最基本的结构。

常见的模型结构有：
- HMM
- CRF
- LSTM+CRF
- CNN+CRF
- BERT+（LSTM）+CRF

【词性标注】[bi-lstm-crf-master](https://github.com/GlassyWing/bi-lstm-crf)
通过Bi-LSTM获得每个词所对应的所有标签的概率，取最大概率的标注即可获得整个标注序列，模型架构变为Embedding + Bi-LSTM + CRF
【分词】[jieba-master](https://github.com/fxsjy/jieba)
“结巴”中文分词：做最好的 Python 中文分词组件
【命名实体识别】[named_entity_recognition](https://github.com/luopeixiang/named_entity_recognition)
中文命名实体识别（包括多种模型：HMM，CRF，BiLSTM，BiLSTM+CRF的具体实现）

#### 主题聚类
文本聚类的主要步骤:
1. 分词
2. 去除停用词
3. 构建词袋空间VSM(vector space model)
4. TF-IDF构建词权重
5. 使用聚类算法进行聚类（KMeans，DBSCAN，BIRCH等）

[TopicCluster-master](https://github.com/liuhuanyong/TopicCluster)包括k-means算法、LDA模型，基于Kmeans与Lda模型的多文档主题聚类,输入多篇文档,输出每个主题的关键词与相应文本,可用于主题发现与热点分析