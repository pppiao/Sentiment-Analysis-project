# -*- coding = utf-8 -*-
"""
@file name : model.py
@author    : tongpiao
@time      : 2020/10/30 15:32
@brief     : 
"""

import sklearn
from sklearn.linear_model import LogisticRegression
import  tensorflow as tf

import tensorflow_datasets
import numpy
import torch


from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification

#TODO extract feature
class FeatureBuilder():
    def __init__(self, method='tfidf'):
        self.method = method



    def get_feature(self, train_data, test_data, tokenizer=None):
        if self.method=="tfidf":
            return self.get_tfidf_feature(train_data, test_data)
        elif self.method=='sentence piece':
            return self.get_bert_feature(train_data, test_data)

    def get_tfidf_feature(train_data, test_data):
        """
        ??? train_data  test_data具体是什么样子？？？
        :param test_data:
        :return:
        zip(*a)与zip(a)相反，理解为解压
        返回值都是元组
        """
        X_train_data, y_train_data = zip(*train_data)
        X_test_data, y_test_data = zip(*train_data)

        # 实例化一个词向量对象？？
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
        """
        fit_transform()先拟合数据，再标准化
        transform()数据标准化  
        为什么训练数据需要先拟合，再标准化？
        而测试数据直接标准化就行了
        """
        # 训练数据的特征
        X_train = vectorizer.fit_transform(X_train_data)
        y_train = vectorizer.fit_transform(y_train_data)
        # 测试数据的特征
        X_test = vectorizer.transform(X_test_data)
        y_test = vectorizer.transform(y_test_data)
        return X_train, y_train, X_test,y_test


    def get_bert_feature(self, train_data, test_data, tokenizer):
        return tokenizer.encode(train_data), tokenizer.encode(test_data)


#TODO 用于二分类
class LinearModel():
    """
    algrithm
    logreg
    logreg_cv
    train
    predict
    """
    def __init__(self):
        self.algrithm = 'LR'
        grid = {"C": numpy.logspace(-3, 3, 7)}
        # logspace用于创建等比数列
        """
        在scikit-learn中，与逻辑回归有关的主要是这3个类。
        LogisticRegression， LogisticRegressionCV 和logistic_regression_path
        其中LogisticRegression和LogisticRegressionCV的主要区别是LogisticRegressionCV使用了交叉验证来选择正则化系数C。而LogisticRegression需要自己每次指定一个正则化系数。
        除了交叉验证，以及选择正则化系数C以外， LogisticRegression和LogisticRegressionCV的使用方法基本相同
        solver 优化算法损失函数
        """
        self.logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.logreg_cv = sklearn.model_selection.GridSearchCV(self.logreg, grid,cv=10, scoring='f1')

    def train(self, X_train, y_train):
        self.logreg_cv.fit(X_train, y_train)
        # buid a text show the main classification metrics
        print(sklearn.metrics.classification_report(y_test, y_pred))

    def predict(self, X_test):
        y_pred = self.logreg_cv.predict(X_test)




# TODO
class NNModel():
    def __init__(self):
        # 对model和tokenizer进行初始化的代码
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = TFBertForSequenceClassification.from_pretained('bert-baes-cased')

    def init_model(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


    def get_tokenizer(self):
        return self.tokenizer


    def train(self, X_train, y_train):
        """

        :param X_train:
        :param y_train:
        :return: 在指定的位置保存训练好的模型
        torch.tensor是一个包含多个同类数据类型数据的多维矩阵。
        典型的tensor构建方法：
        torch.tensor(data, dtype=None, device=None, requires_grad=False)
        """
        input_ids = torch.tensor(X_train)
        history = self.model.fit(input_ids, epochs=2, steps_per_epoch=115, \
                                 validation_data=valid_dataset, validation_steps=7)
        self.model.save_pretrained('./save/')


    def predict(self, X_test):

        return self.model(torch.tensor(X_test)).argmax().item()