import numpy as np

def accuracy_score(y_true, y_predict):
    """計算y_true和y_predict之間的準確率"""
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_ture must be equal to the size of y_predict"

    return sum(y_true == y_predict) / len(y_true)