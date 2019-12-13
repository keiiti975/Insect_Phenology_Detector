# Insect Phenology Detector  
ホワイトボードに映った昆虫の個体数を計測するためのシステム  

### --- TODO ---  
### 実験関連  
- [ ] compare_div_model.ipynbの完成  
    - [x] (div + cls)resnet101_b20_r45_lr1e-5_crossvalid_divide_resizeの実験を回す  
    - [x] (div + cls)resnet50_b20_r45_lr1e-5_crossvalid_resizeの実験を回す  
    - [x] (cls only)resnet50_b20_r45_lr1e-5_crossvalid_resize_otherの実験を回す  
    - [ ] (cls only without grouping)resnet50_b20_r45_lr1e-5_crossvalid_resize_other_without_groupingの実験を回す  
- [ ] compare_models.ipynbの完成  
    - [x] ResNet101/resnet18_b20_r45_lr1e-5_crossvalid_not_pretrainの実験を回す  
    - [x] ResNet101/resnet18_b20_r45_lr1e-5_crossvalidの実験を回す  
    - [x] ResNet101/resnet34_b20_r45_lr1e-5_crossvalid_not_pretrainの実験を回す  
    - [ ] ResNet101/resnet34_b20_r45_lr1e-5_crossvalidの実験を回す  
    - [ ] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_not_pretrainの実験を回す  
    - [ ] ResNet101/resnet50_b20_r45_lr1e-5_crossvalidの実験を回す  
    - [ ] ResNet101/resnet101_b20_r45_lr1e-5_crossvalid_not_pretrainの実験を回す  
    - [ ] ResNet101/resnet101_b20_r45_lr1e-5_crossvalidの実験を回す  
- [ ] compare_insect_resize.ipynbの完成  
    - [ ] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_resizeの実験を回す  
- [ ] compare_DCL.ipynbの完成  
    - [ ] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_resize_DCLの実験を回す  

### コード修正  
- [ ] train_ResNet.ipynbのtrain()をcheckpointごとに保存可能にする  
- [ ] train_RefineDet.ipynbのtrain()にコスト考慮型学習を適用する  

### その他
- 30分、1時間ずっと同じ場所にいる昆虫をゴミかどうか判断して、ゴミなら背景差分を取って取り除く  
- 昆虫の体長を測れるモデルの構築  
- 昆虫の分類形質を学習時に意味表現として与える  

---  
### 研究ログ  
- 2019/12  
    - [x] compare_crop.ipynbの完成  
        - [x] RefineDet/b2_2_4_8_16_32_im512の実験を回す  
    - 検出結果から水生昆虫を分離するモデルの比較  
    結果の場所: det2cls/compare_div_model  
    →水生昆虫判別器を使用しないほうが結果が良くなった(AP 0.654 vs 0.786)  

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