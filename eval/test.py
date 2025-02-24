import jieba

text = "分分合合"
words = list(jieba.cut(text))
print(words) 

text = "我明白了了，也懂得了分分合合"
words = list(jieba.cut(text))
print(words) 

text = "“我明白了了，也懂得了分分合合”"
words = list(jieba.cut(text))
print(words) 

text = "“我明明白白明白了了，也懂得了分分合合”"
words = list(jieba.cut(text))
print(words) 

text = "荒野里狼眼闪着绿光，一下一下，一下子我清醒了过来"
words = list(jieba.cut(text))
print(words) 

text = "山东最高的山是山连山、山连山、山连山"
words = list(jieba.cut(text))
print(words) 

text = "我明白了了，小时了了，大未必佳，不了了之，了了解了吗"
words = list(jieba.cut(text))
print(words) 

text = "啦啦啦，啦啦啦，我是卖报的小行家，嘻嘻哈哈"
words = list(jieba.cut(text))
print(words) 