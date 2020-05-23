### 各モデルの使用方法  
### --- Faster-RCNN ---  
- 必要ライブラリ  
pytorch >= 1.0  
torchvision >= 0.3  
- 読み込みデータ形式  
クラスラベルは1始まり(背景クラス:0、識別クラス:1,2,...)  
[x1:Float, y1:Float, x2:Float, y2:Float, label: Integer]の形で読み込む  
(x1, y1)はボックスの左上の座標、(x2, y2)はボックスの右下の座標  
- モデル入力時のデータ形式  
targets = {"boxes": torch.from_numpy(np.asarray([[x1, y1, x2, y2], ...])),  
"labels": torch.from_numpy(np.asarray([label, ...]))}の形の辞書を教師データとして与える  
boxesとlabelsの入れる順番は合わせておく  
座標は画素の位置(0\~H_max or 0\~W_max)  

### --- RefineDet ---  
- 必要ライブラリ  
pytorch >= 1.0  
- 読み込みデータ形式  
クラスラベルは0始まり(背景クラス:-1、識別クラス:0,1,...)  
[x1:Float, y1:Float, x2:Float, y2:Float, label: Float]の形で読み込む  
(x1, y1)はボックスの左上の座標、(x2, y2)はボックスの右下の座標  
- モデル入力時のデータ形式  
targets = torch.from_numpy(np.asarray([[x1, y1, x2, y2, label], ...]))を教師データとして与える  
座標は0.0~1.0に正規化して渡す必要がある  
識別時は1始まり(学習時にラベルが+1されて、背景クラスが0になるように補正される)  
refine_match()の中にラベル補正箇所あり  

### --- ResNet ---  
- 必要ライブラリ  
pytorch >= 0.4  
torchvision >= 0.2  
- モデル入力時のデータ形式  
画像: 200[pixel] * 200[pixel]  
教師データ: label  