# Insect Phenology Detector  
ホワイトボードに映った昆虫の個体数を計測するためのシステム  

### --- TODO ---  
### 実験関連  
- [ ] compare_cls.ipynbの完成  
    - [ ] RefineDet/crop_b2_2_4_8_16_32_im512_clsの実験を回す  

### コード修正  
- [ ] train_RefineDet.ipynbのtrain()にコスト考慮型学習を適用する  
- [ ] train_RefineDet.ipynbをフルHDで学習可能にする  
- [ ] RefineDetで検出と分類を同時に学習出来るようにする  

### その他  
- 30分、1時間ずっと同じ場所にいる昆虫をゴミかどうか判断して、ゴミなら背景差分を取って取り除く  
- 昆虫の体長を測れるモデルの構築  
- 昆虫の分類形質を学習時に意味表現として与える  

---  
### 研究ログ  
- 2019/12  
    - [ ] train_ResNet.ipynbのtrain()をcheckpointごとに保存可能にする  
    - [x] compare_crop.ipynbの完成  
        - [x] RefineDet/b2_2_4_8_16_32_im512の実験を回す  
    - [ ] compare_CSL.ipynbの完成  
        - [x] RefineDet/crop_b2_2_4_8_16_32_im512_CSL_param1の実験を回す  
        - [ ] RefineDet/crop_b2_2_4_8_16_32_im512_CSL_param2の実験を回す  
    - [x] compare_models.ipynbの完成  
        - [x] ResNet101/resnet18_b20_r45_lr1e-5_crossvalid_not_pretrainの実験を回す  
        - [x] ResNet101/resnet18_b20_r45_lr1e-5_crossvalidの実験を回す  
        - [x] ResNet101/resnet34_b20_r45_lr1e-5_crossvalid_not_pretrainの実験を回す  
        - [x] ResNet101/resnet34_b20_r45_lr1e-5_crossvalidの実験を回す  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_not_pretrainの実験を回す  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalidの実験を回す  
        - [x] ResNet101/resnet101_b20_r45_lr1e-5_crossvalid_not_pretrainの実験を回す  
        - [x] ResNet101/resnet101_b20_r45_lr1e-5_crossvalidの実験を回す  
    - [x] compare_div_model.ipynbの完成  
        - [x] (div + cls)resnet101_b20_r45_lr1e-5_crossvalid_divide_resizeの実験を回す  
        - [x] (div + cls)resnet50_b20_r45_lr1e-5_crossvalid_resizeの実験を回す  
        - [x] (cls only)resnet50_b20_r45_lr1e-5_crossvalid_resize_otherの実験を回す  
        - [x] (cls only without grouping)resnet50_b20_r45_lr1e-5_crossvalid_resize_other_without_groupingの実験を回す  
    - [x] compare_insect_resize.ipynbの完成  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_resizeの実験を回す  
    - [x] compare_DCL.ipynbの完成  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_resize_DCLの実験を回す  
    - 検出結果から水生昆虫を分離するモデルの比較  
    結果の場所: det2cls/compare_div_model  
    →水生昆虫判別器を使用しないほうが結果が良くなった(AP 0.654 vs 0.786)  
    また分類モデルは全ての昆虫の分類を学習してから水生昆虫とその他の昆虫に分けた方が結果が良くなった(AP 0.786 vs 0.826)  
    - 検出モデルのコスト考慮型学習の学習重みの考察  
    ||target negative|target positive|  
    |:-:|:-:|:-:|  
    |output negative|1|0.5|  
    |output positive|1.5|1|  
    - RefineDetのコスト考慮型学習(CSL,Cost-Sensitive Learning)  
    学習誤差がクロスエントロピーで与えられるため、クラス別の重みしか定義できない  
        - [1.2, 0.8]で学習してみる(param1)  
        結果の場所: detection/compare_crop  
        →結果は良くならなかった  
        - [0.8, 1.2]で学習してみる(param2)  
        結果の場所: detection/compare_crop  
        →結果は良くならなかった  
    - データセット:classify_insect_std_resize_aquatic_other_without_groupingの作成  
    {'Diptera': 0, 'Ephemeridae': 1, 'Ephemeroptera': 2, 'Lepidoptera': 3, 'Plecoptera': 4  
    , 'Trichoptera': 5, 'Coleoptera': 6, 'Hemiptera': 7, 'medium insect': 8, 'small insect': 9}  
    - 4k画像をそもそも使う必要があるのか?  
    検出モデルをフルHDで学習してみる、tristanさん曰く精度は良くならない(ただ学習時間は短くなる)  
    - 検出モデルで分類も学習してみる  
    クラスは全部で13個(背景クラス+昆虫クラス)ある  
    - 分類モデルで昆虫画像をresizeしたものとしてないものを比較  
    結果の場所: classification/compare_insect_size  
    ほとんど同じ結果、実は昆虫の大きさも学習の重要な要因なのかも...  
    - 分類モデルでDCLを使用したものとしてないものを比較  
    結果の場所: classification/compare_DCL  
    小さい昆虫の識別率が悪くなっただけ、今回のタスクでは大きいモデルの学習が難しいのかもしれない  

---  
### 昆虫の分類形質  
- カゲロウ  
色はほぼ同じ  
頭大きい  
尾が長い、羽が透明  
止まるときはハエが垂直で、かつ頭が大きく見える  
- カワゲラ  
色はほぼ同じ  
頭小さい  
左右の羽をくっつけて畳む、羽の部分が広く見える  
- トビケラ  
色はほぼ同じ  
羽を三角形に畳む、棒のように見える  
- ハエ  
足が6本しっかり見える。羽が横に垂直に開く  
- チョウ(ガ)  
色は様々  