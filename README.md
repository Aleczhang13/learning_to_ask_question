#### 2019年08月24日
学习了torchtext在数据上如何构建examples和如何构建词表等方式
使用config进行控制试验参数
完成了初步的encoder的代码

#### 2019年08月25日
完成的简单的生成，尝试增加attention, copy network 以及point network

#### 2019年08月25日
增加了attention模块

#### 2019年08月26日
（1）发现关于context的句子长度是非常重要的，这是一个需要重点调控的参数  
（2）让glove可以进行finetune在dev上测试集可以达到较好的acc，但是生成的句子的BLEU反而降了一个点

#### 2019年09月09日
（1）确定了torchtext中max-size没有bug