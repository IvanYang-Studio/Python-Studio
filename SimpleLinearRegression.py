import numpy as np
from .metrics import r2_score


class SimpleLinearRegression:

    def __init__(self):
        """初始化 Simple Linear Regression 模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根據訓練數據集 x_train , y_train 訓練 Simple Linear Regression 模型"""
        assert x_train.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data"
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)
        """向量方式運算"""

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """給定特定預測數據集 x_predict  ，返回表示 x _predict 的結果向量"""
        assert x_predict.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """給訂單個待預測數據 x_single ，返回 x_single 的預測結果值"""
        return self.a_ * x_single + self.b_

    def score(self, x_test, y_test):
        """根據測試數據集 x_test 和 y_test 確定當前模型的準確度"""

        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression()"


'''
class SimpleLinearRegression1:

    def __init__(self):
        """初始化 Simple Linear Regression 模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根據訓練數據集 x_train , y_train 訓練 Simple Linear Regression 模型"""
        assert x_train.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data"
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0
        for x, y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """給定特定預測數據集 x_predict  ，返回表示 x _predict 的結果向量"""
        assert x_predict.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """給訂單個待預測數據 x_single ，返回 x_single 的預測結果值"""
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"
'''