# -*- coding: utf-8 -*-
import numpy as np
import visdom

def visualize(vis, phase, visualized_data, window):
    """
        Visdomの可視化関数
        引数:
            - vis: visdom.Visdom, visdomクラス
            - phase: int, 現在のエポック
            - visualized_data: float, 誤差や正答率などの可視化したいデータ
            - window: visdom.Visdom.line, 窓クラス
    """
    vis.line(
        X=np.array([phase]),
        Y=np.array([visualized_data]),
        update='append',
        win=window
    )