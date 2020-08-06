这里整理一些知名的NLP开源项目，分别为：

#### 词义消歧
[WordMultiSenseDisambiguation](https://github.com/liuhuanyong/WordMultiSenseDisambiguation)
WordMultiSenseDisambiguation, chinese multi-wordsense disambiguation based on online bake knowledge base and semantic embedding similarity compute,基于百科知识库的中文词语多义项获取与特定句子词语语义消歧. 

#### 拼写纠错
[bert_chinese](https://github.com/JohanyCheung/bert_chinese/tree/master/corrector)
直接预训练的bert模型实现中文的文本纠错，可参照学习bert如何做纠错任务
[FASPell](https://github.com/iqiyi/FASPell/blob/master)
使用bert进行预训练+微调，再经过CSD过滤器得到最终结果。支持简体中文文本； 繁体中文文本； 人类论文； OCR结果等
[pycorrector](https://github.com/shibing624/pycorrector)
Pycorrector：当前主流的中文纠错框架，支持规则和端到端模型
[SoftMaskedBert](https://github.com/hiyoung123/SoftMaskedBert)
对论文Soft-Masked Bert 的复现，使用判别模型BiGRU+纠错模型BERT，实现端到端的纠错。
[YoungCorrector](https://github.com/hiyoung123/YoungCorrector)
基于Pycorrector改造，实现基于纯规则的纠错系统。整个系统框架比较详细。与Pycorrector准确度差不多，耗时短（归功于前向最大匹配替代了直接索引混淆词典）

#### 文本分类
[Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)
中文文本分类，TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention, DPCNN, Transformer, 基于pytorch，开箱即用。
[Naive-Bayes-Classifier](https://github.com/lining0806/Naive-Bayes-Classifier)
使用朴素贝叶斯进行文本分类：
[text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn)
使用CNN以及RNN进行中文文本分类
[text-cnn](https://github.com/cjymz886/text-cnn)
使用CNN模型进行文本分类

#### 序列标注
【词性标注】[bi-lstm-crf](https://github.com/GlassyWing/bi-lstm-crf)
通过Bi-LSTM获得每个词所对应的所有标签的概率，取最大概率的标注即可获得整个标注序列，模型架构变为Embedding + Bi-LSTM + CRF
【分词】[jieba](https://github.com/fxsjy/jieba)
“结巴”中文分词：做最好的 Python 中文分词组件
【命名实体识别】[named_entity_recognition](https://github.com/luopeixiang/named_entity_recognition)
中文命名实体识别（包括多种模型：HMM，CRF，BiLSTM，BiLSTM+CRF的具体实现）

#### 主题聚类
包括k-means算法、LDA模型：[TopicCluster](https://github.com/liuhuanyong/TopicCluster)
基于Kmeans与Lda模型的多文档主题聚类,输入多篇文档,输出每个主题的关键词与相应文本,可用于主题发现与热点分析