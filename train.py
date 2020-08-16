# coding: utf-8
from __future__ import unicode_literals

from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dense

VOCAB_SIZE = 2000
EMBEDDING_DIM = 100
MAX_WORDS = 500
CLASS_NUM = 5


def build_fastText():
    model = Sequential()
    # 将词汇数VOCAB_SIZE映射为EMBEDDING_DIM维
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_WORDS))
    # 平均文档中所有词的embedding
    model.add(GlobalAveragePooling1D())
    # softmax分类
    model.add(Dense(CLASS_NUM, activation='softmax'))
    # 定义损失函数、优化器、分类度量指标
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    model = build_fastText()
    print(model.summary())


import time
import numpy as np
import fasttext
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

def fasttext_model(nrows, train_num, lr=1.0, wordNgrams=2, minCount=1, epoch=25, loss='hs', dim=100):
    start_time = time.time()

    # 转换为FastText需要的格式
    train_df = pd.read_csv('input/train_set.csv', sep='\t', nrows=nrows)

    # shuffle
    train_df = shuffle(train_df, random_state=666)

    train_df['label_ft'] = '__label__' + train_df['label'].astype('str')
    train_df[['text', 'label_ft']].iloc[:train_num].to_csv('input/fastText_train.csv', index=None, header=None, sep='\t')

    model = fasttext.train_supervised('input/fastText_train.csv', lr=lr, wordNgrams=wordNgrams, verbose=2,
                                      minCount=minCount, epoch=epoch, loss=loss, dim=dim)

    train_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[:train_num]['text']]
    print('Train f1_score:', f1_score(train_df['label'].values[:train_num].astype(str), train_pred, average='macro'))
    val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[train_num:]['text']]
    print('Val f1_score:', f1_score(train_df['label'].values[train_num:].astype(str), val_pred, average='macro'))
    train_time = time.time()
    print('Train time: {:.2f}s'.format(train_time - start_time))

     # 预测并保存
    test_df = pd.read_csv('input/test_a.csv')

    test_pred = [model.predict(x)[0][0].split('__')[-1] for x in test_df['text']]
    test_pred = pd.DataFrame(test_pred, columns=['label'])
    test_pred.to_csv('input/test_fastText_ridgeclassifier.csv', index=False)
    print('Test predict saved.')
    end_time = time.time()
    print('Predict time:{:.2f}s'.format(end_time - train_time))


if __name__ == '__main__':
    nrows = 200000
    train_num = int(nrows * 0.7)
    lr=0.01
    wordNgrams=2
    minCount=1
    epoch=25
    loss='hs'

    fasttext_model(nrows, train_num)