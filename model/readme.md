### 各モデルの使用方法  
### --- Faster-RCNN ---  
研究で使ってないので使えないかも...  
- 必要ライブラリ  
pytorch >= 1.0  
torchvision >= 0.3  
- 読み込みデータ形式  
クラスラベルは1始まり(背景クラス:0、識別クラス:1,2,...)で与える  
[x1:Float, y1:Float, x2:Float, y2:Float, label: Integer]の形で読み込む  
(x1, y1)はボックスの左上の座標、(x2, y2)はボックスの右下の座標  
- モデル入力時のデータ形式  
画像: 任意の大きさのRGB画像  
教師データ: targets = {"boxes": torch.Tensor([[x1, y1, x2, y2], ...]), "labels": torch.Tensor([label, ...])}の形の辞書を教師データとして与える  
boxesとlabelsの入れる順番は合わせておく <-- 注意!!  
座標は画素の位置(0\~H_max or 0\~W_max)  

### --- RefineDet ---  
- 必要ライブラリ  
pytorch >= 1.0  
- 読み込みデータ形式  
クラスラベルは0始まり(背景クラス:-1、識別クラス:0,1,...)で与える  
[x1:Float, y1:Float, x2:Float, y2:Float, label: Float]の形で読み込む  
(x1, y1)はボックスの左上の座標、(x2, y2)はボックスの右下の座標  
- モデル入力時のデータ形式  
画像: 512[pixel] * 512[pixel]のRGB画像(256 * 256, 1024 * 1024でも可)  
教師データ: targets = torch.Tensor([[x1, y1, x2, y2, label], ...])を教師データとして与える  
座標は0.0~1.0に正規化して渡す必要がある  
テスト時の識別は1始まり(学習時にラベルが+1されて、背景クラスが0になるように補正されている) <-- 注意!!  
refine_match()の中にラベル補正箇所あり  

### --- ResNet, MobileNet ---  
- 必要ライブラリ  
pytorch >= 1.0  
torchvision >= 0.2  
- モデル入力時のデータ形式  
配列はtorch.Tensor型  
画像: 指定なし  
教師データ: label, dtype=int  

### --- SegNet, Unet ---  
研究で使ってない(以前の研究のもの)  
- 必要ライブラリ  
無し  
- モデル入力時のデータ形式  
配列はtorch.Tensor型  
画像: 指定なし  
教師データ: segmentation_map, 入力画像と同サイズのグレースケール画像  