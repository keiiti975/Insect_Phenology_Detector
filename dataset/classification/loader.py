# -*- coding: utf-8 -*-
import numpy as np


def create_validation_split(Y, test_ratio=0.2):
    """
        学習/テストの交差検証用インデックスを作成
        引数:
            - Y: np.array, size=[insect_num], 昆虫ラベル
            - test_ratio: float, テストに使用するインデックスの割合
            test_ratioから交差検証の回数を計算している
    """
    idx, counts = np.unique(Y, return_counts=True)
    ntests = counts * test_ratio
    ntests = np.round(ntests).astype("int")

    valid_num = int(1.0 / test_ratio)

    valid_idx = [np.where(Y == i)[0] for i in range(len(ntests))]
    default_idx = [np.where(Y == i)[0] for i in range(len(ntests))]

    test = []
    train = []
    for i in range(valid_num):
        if i == valid_num - 1:
            ntests = counts - ntests * (valid_num - 1)

        valid_test = []
        valid_train = []
        for i, n in enumerate(ntests):
            test_id = np.asarray(valid_idx[i][:n]) # クラス別のテストインデックスを取得
            train_id = list(set(default_idx[i]) - set(test_id)) # クラス別の学習インデックスを取得
            valid_test.extend(test_id.tolist()) # テストインデックスを交差検証用インデックスの配列に格納
            valid_train.extend(train_id) # 学習インデックスを交差検証用インデックスの配列に格納
            valid_idx[i] = list(set(valid_idx[i]) - set(test_id)) # 次の交差検証インデックス作成のために、テストインデックスを除去
        test.append(valid_test)
        train.append(valid_train)

    return train, test


def load_validation_data(X, Y, valid_train_idx, valid_test_idx):
    """
        学習/テストデータをインデックスから読み込む
        Args:
            - X: np.array, 昆虫画像全体
            - Y: np.array, ラベル全体
            - valid_train_idx: np.array, 交差検証用学習インデックス
            - valid_test_idx: np.array, 交差検証用テストインデックス
    """
    xtr = X[valid_train_idx]
    ytr = Y[valid_train_idx]
    xte = X[valid_test_idx]
    yte = Y[valid_test_idx]
    return xtr, ytr, xte, yte