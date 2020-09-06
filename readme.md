# Insect Phenology Detector  
ホワイトボードに映った昆虫の個体数を計測するためのシステム  

### --- TODO ---  
### 実験関連  

### コード修正  
- [ ] train_RefineDet.ipynbをフルHDで学習可能にする  
- [ ] 昆虫の体サイズを予測できるようにモデルを修正  
- [ ] 分類モデルで識別を間違えた昆虫のラベルを出力  

### その他  
- 30分、1時間ずっと同じ場所にいる昆虫をゴミかどうか判断して、ゴミなら背景差分を取って取り除く  
- 昆虫の体長を測れるモデルの構築  
- 昆虫の分類形質を学習時に意味表現として与える  

---  
### 研究ログ  
- 2019/12  
    - [x] train_ResNet.ipynbのtrain()をcheckpointごとに保存可能にする  
    - [x] compare_crop.ipynbの完成  
        - [x] RefineDet/b2_2_4_8_16_32_im512の実験を回す  
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
    - [x] compare_CSL.ipynbの完成  
        - [x] RefineDet/crop_b2_2_4_8_16_32_im512_CSL_param1の実験を回す  
        - [x] RefineDet/crop_b2_2_4_8_16_32_im512_CSL_param2の実験を回す  
    - [x] detection_with_cls.ipynbの完成  
        - [x] RefineDet/crop_b2_2_4_8_16_32_im512_clsの実験を回す  
    - 検出結果から水生昆虫を分離するモデルの比較  
    結果の場所: det2cls/compare_div_model  
    →水生昆虫判別器を使用しないほうが結果が良くなった(AP 0.654 vs 0.786)  
    また分類モデルは全ての昆虫の分類を学習してから水生昆虫とその他の昆虫に分けた方が結果が良くなった(AP 0.786 vs 0.826)  
    - 検出モデルのコスト考慮型学習の学習重みの考察  
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
    - 検出モデルで事前学習したときに小さい昆虫の識別率だけ悪くなる  
    事前学習のデータセットには小さい物体が含まれていない可能性がある  
    →小さい物体を含めた事前学習の方法を提案できないか?  
    →小さい物体の検出を学習しやすくする方法はないか?  
    - 分類モデルの学習時に個体数の多い大きさで識別率が悪くなる  
    個体数が多いことで学習が難しくなっている、昆虫同士が似ているため判別できない  
    →個体数を減らすのは良くないので、個体数の少ない大きさの昆虫をupsamplingする  
    →昆虫の局所的な特徴を学習しやすくする方法を考える  
- 2020/1  
    - [x] compare_finetuning(det).ipynbの完成  
        - [x] RefineDet/crop_b2_2_4_8_16_32_im512_not_pretrainの実験を回す  
        - [x] RefineDet/crop_b2_2_4_8_16_32_im512の実験を回す  
        - [x] RefineDet/crop_b2_2_4_8_16_32_im512_freezeの実験を回す  
    - [x] compare_finetuning(cls).ipynbの完成  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_not_pretrainの実験を回す  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalidの実験を回す  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_freezeの実験を回す  
    - imbalanced-learnを用いてUndersampling/OversamplingをするのにPCAが必須  
    - RandomSizeCropでは物体を小さく出来ないので、物体を小さくする方法を考える必要あり  
    - テスト結果に応じて学習量を変えれば、少し結果が良くなった。  
- 2020/2  
    - [x] compare_correction.ipynbの完成  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalidの実験を回す  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_correctionの実験を回す  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_correction_LowTrainableの実験を回す  
    - [x] compare_resize.ipynbの完成  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalidの実験を回す  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_resizeの実験を回す  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_resize_crossvalidの実験を回す  
    - [x] compare_sampling.ipynbの完成  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalidの実験を回す  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_randomsamplingの実験を回す  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_randomoversamplingの実験を回す  
    - Zero-Paddingで学習した分類モデルの中間特徴量がおかしい  
    →分類モデルの画像を大きさを揃える処理はZero-PaddingよりResizeの方が良い  
    - 水生昆虫以外が入っているデータセットを削除  
    - 検出モデルでは昆虫の種類ごとにIoUの値を変えても良いのでは  
    - 分類モデルでは過学習していることが分かった。学習する特徴を増やせるかも...  
    - 分類モデルで入力サイズを揃える処理はZero-PaddingよりResizeの方が良さそう  
    結果は同程度だが、特徴量はResizeの方が正しく得られていた  
    - 分類モデルでRegionConfusionMechanism単体のデータ拡張をすると結果が良くなった  
    - 入力寄りの中間特徴量しか識別に関与していないかもしれない  
- 2020/3  
    - [x] compare_decoder.ipynbの完成  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalidの実験を回す  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_concatenateの実験を回す  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_FPNの実験を回す  
    - [x] compare_concatenate_resize.ipynbの完成  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_concatenateの実験を回す  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_resizeFAR_concatenateの実験を回す(FAR=Fix Aspect Ratio)  
        - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_resize_concatenateの実験を回す  
    - oversamplingは結果良くなりそう  
    - ResNet_concatenateは結果良くなる  
- 2020/6  
    - [x] train_RefineDet.ipynbのtrain()にコスト考慮型学習を適用する  
    - [x] RefineDetで検出と分類を同時に学習出来るようにする  
    - [ ] compare_divide_architecture.ipynbの完成  
        - refinedet_det2cls  
            - [x] RefineDet/crop_b2_2_4_8_16_32_im512_det2clsの実験を回す  
        - refinedet_resnet_plus_other  
            - [x] RefineDet/crop_b2_2_4_8_16_32_im512の実験を回す  
            - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_otherの実験を回す  
        - refinedet_plus_other_resnet  
            - [x] RefineDet/crop_b2_2_4_8_16_32_im512_otherの実験を回す  
            - [x] ResNet101/resnet50_b20_r45_lr1e-5_crossvalidの実験を回す  
    - 検出モデルはGroup Normalization+Weight Standardizationで結果が少し改善  
    - クラス内サイズ分散とクラス別識別率・サイズ別識別率の関係を調べたが、有意差はなかった  
    - クラス内サイズ分散について気になった点  
        - 個体数が多いとresizeFARが有利  
        - 昆虫サイズが大きいとconcat無しの方が良い  
- 2020/7  
    - [x] 検出評価コードのAPが違う原因を調べる  
        - refinedetのtop_kを100から1000に変更すると同等の結果が出た  
        - 100epochだと過学習しているので、20epochで評価を行う必要がある  
    - [x] compare_autoaugment_oversample.ipynbの完成  
        - [x] ResNet101/resnet50_b20_lr1e-5_crossvalid_fastautoaugmentの実験を回す  
        - [x] ResNet101/resnet50_b20_lr1e-5_crossvalid_fastautoaugment_randomoversampleの実験を回す  
    - 以下のデータセットを作成  
        - classify_insect_std_plus_other(分類)  
        - refinedet_plus_other(refinedet)  
        - target_with_other(det2clsの評価用)  
    - 分類モデルにautoaugmentを適用  
    昆虫認識とautoaugmentは相性が悪そう  
    - クラス別の個体数を揃えるためにOverSampleが便利  
    - 画像データを偏りをFew-Shot学習の視点から解決していこう  
    (Class-Imbalanceの解決を目指す論文(関連研究)がそもそも少ない)  
    - 辞書のコピーは値渡しでないので、copyモジュールを使用する  
    - ["annotations_4"]には"Hemiptera"が含まれていないので注意!  
    - 検出で使用している評価コードによる結果が、以前の結果と異なる問題が発生  
- 2020/8  
    - [x] 佐藤先生に新しくもらった画像データのモデル出力を送る  
    - [ ] visualize_annotation_20200806.ipynbの完成  
    - [x] 検出評価コードの検出結果を解析するコードの実装  
    - [ ] compare_add_data.ipynbの完成  
        - [ ] RefineDet/crop_b2_2_4_8_16_32_im512の実験を回す  
        - [ ] RefineDet/crop_b2_2_4_8_16_32_im512_20200806の実験を回す  
    - [ ] compare_add_data.ipynbの完成  
        - [ ] ResNet101/resnet50_b20_r45_lr1e-5_crossvalidの実験を回す  
        - [ ] ResNet101/resnet50_b20_r45_lr1e-5_crossvalid_20200806の実験を回す  
    - [ ] compare_augmentations.ipynbの完成  
        - [ ] Shear: crop_b2_2_4_8_16_32_im512_20200806_shear  
        - [ ] Translate: crop_b2_2_4_8_16_32_im512_20200806_translate  
        - [ ] Rotate: crop_b2_2_4_8_16_32_im512_20200806_rotate  
        - [ ] AutoContrast: crop_b2_2_4_8_16_32_im512_20200806_autocontrast  
        - [ ] Invert: crop_b2_2_4_8_16_32_im512_20200806_invert  
        - [ ] Equalize: crop_b2_2_4_8_16_32_im512_20200806_equalize  
        - [ ] Solarize: crop_b2_2_4_8_16_32_im512_20200806_solarize  
        - [ ] Posterize: crop_b2_2_4_8_16_32_im512_20200806_posterize  
        - [ ] Contrast: crop_b2_2_4_8_16_32_im512_20200806_contrast  
        - [ ] Color: crop_b2_2_4_8_16_32_im512_20200806_color  
        - [ ] Brightness: crop_b2_2_4_8_16_32_im512_20200806_brightness  
        - [ ] Sharpness: crop_b2_2_4_8_16_32_im512_20200806_sharpness  
        - [ ] Cutout: crop_b2_2_4_8_16_32_im512_20200806_cutout  
    - [ ] compare_augmentations.ipynbの完成  
        - [ ] Shear: resnet50_b20_r45_lr1e-5_crossvalid_20200806_shear  
        - [ ] Translate: resnet50_b20_r45_lr1e-5_crossvalid_20200806_translate  
        - [ ] Rotate: resnet50_b20_r45_lr1e-5_crossvalid_20200806  
        - [ ] AutoContrast: resnet50_b20_r45_lr1e-5_crossvalid_20200806_autocontrast  
        - [ ] Invert: resnet50_b20_r45_lr1e-5_crossvalid_20200806_invert  
        - [ ] Equalize: resnet50_b20_r45_lr1e-5_crossvalid_20200806_equalize  
        - [ ] Solarize: resnet50_b20_r45_lr1e-5_crossvalid_20200806_solarize  
        - [ ] Posterize: resnet50_b20_r45_lr1e-5_crossvalid_20200806_posterize  
        - [ ] Contrast: resnet50_b20_r45_lr1e-5_crossvalid_20200806_contrast  
        - [ ] Color: resnet50_b20_r45_lr1e-5_crossvalid_20200806_color  
        - [ ] Brightness: resnet50_b20_r45_lr1e-5_crossvalid_20200806_brightness  
        - [ ] Sharpness: resnet50_b20_r45_lr1e-5_crossvalid_20200806_sharpness  
        - [ ] Cutout: resnet50_b20_r45_lr1e-5_crossvalid_20200806_cutout  
    - データ拡張について古典的なアプローチ(全数調査など)で調査する  
    - 以下のデータセットを作成  
        - refinedet_all_20200806(refinedet)  
        - refinedet_plus_other_20200806(refinedet)  
        - refinedet_all_test_20200806(refinedet)  
        - classify_insect_std_20200806(分類)  
        - classify_insect_only_20200806(分類)  
    - 検出にimgaugを用いたデータ拡張を実装  
        - SPREAD_ALL_OVERの関数が使用できない(学習時とテスト時で処理される画像が異なる)  
    - 分類にimgaugを用いたデータ拡張を実装  
    - 検出にはRotateを使用しない方が良さそう  
    - 検出の結果を修正し、resultも修正  
    - AutoAugmentと同様のデータ拡張を一つずつ実験  
    - 検出でRandomの関数を使用すると、検出率が悪化した  
    - 分類はrotate+translateの組み合わせが良さそう(結果は良くなってない)  
- 2020/9  
    - 研究室発表会での指摘  
        - 検出精度が下がったのはアンカーサイズの問題では?  
        - Dipteraに識別が偏っているのは、データ量の問題  
        - 撮影環境の影響を軽減するために、色々な撮影環境でデータを作れば良い  
        - フェノロジーの話を研究資料に足しておくように  
        - データ拡張が良くなかったのはパラメータの問題があるかも  
    - 追加データのアノテーションに余白が含まれていて、正しく学習・テストが行えない  
        - 佐藤先生にアノテーションの修正をお願い  
        - 追加データのサイズを自力で修正  

---  
### 昆虫の分類形質  
- Ephemeroptera(カゲロウ目)  
###佐藤先生から聞いた特徴###  
色はほぼ同じ、頭が大きい、尾が長い、羽が透明、止まるときは羽が垂直で、かつ頭が大きく見える。  
###日本産水生昆虫###  
亜成虫と呼ばれる蛹に該当する発育段階があり、成虫と見分けがつかない。  
目に複眼がある。中胸部、後胸部にそれぞれ前翅、後翅が発達するが、あったりなかったりする。  
背中に線が入っている。  
- Plecoptera(カワゲラ目)  
###佐藤先生から聞いた特徴###  
色はほぼ同じ、頭が小さい、羽をくっつけて畳む、羽の部分が広く見える。  
###日本産水生昆虫###  
翅を背面に重ねて水平に畳む。頭に触角、複眼を持つ。尾は短い。  
- Trichoptera(トビケラ目)  
###佐藤先生から聞いた特徴###  
色はほぼ同じ、羽を三角形に畳む、棒のように見える。  
###日本産水生昆虫###  
全体的に蛾に似ているが、翅には鱗粉ではなく小毛がある場合が多い。  
触角は棒状。翅は楕円もしくは逆三角形で地味な色が多い。  
- Lepidoptera(チョウ、ガ)  
###佐藤先生から聞いた特徴###  
色は様々  
###日本産水生昆虫###  
特になし  
- Diptera(ハエ目)  
###佐藤先生から聞いた特徴###  
足が6本しっかり見える。羽が横に垂直に開く  
###日本産水生昆虫###  
種類がすごく多い、2枚の翅を持っている。(前翅、後翅)  
足がしっかりしていて、翅も丈夫そう。  