### 引き継ぎ資料  

#### 各フォルダとファイルの説明  
- data  
    - all_classification_data  
    分類モデルを学習するようのデータ。  
    make_classification_data.ipynbを使って作成できる。  
    詳しくはdataset.mdに記載している。  
    - annotations_0, annotations_2, annotations_3, annotations_4, annotations_20200806  
    データセットを作成するためのアノテーションデータ。  
    LabelImgを用いて作成してもらっている。  
    ファイル名は画像と連動している。  
    annotations_0〜annotations_4とannotations_20200806は撮影時期が異なり、  
    annotations_20200806の方が新しいデータである。  
    データ形式はPascalVocのアノテーションと同一のものを使用している。  
    (これらのデータはLabelImgで可視化できるのでぜひとも見て欲しい)  
    - insect_knowledge  
    分類モデルの予測のクラス平均をとったもの。  
    知識蒸留の手法として、学習済みモデルの予測をターゲットラベルに使用する手法があり、  
    それを実装するために作った。  
    - ooe_pict  
    撮影されたが、アノテーションが付けられていない画像。  
    モデル予測を提供するためにあり、相手からモデル予測を出して欲しいと言われたら、  
    適当にフォルダを作ってモデル予測を提供して欲しい。  
    - refined_images  
    学習・テストに使用できる全画像(annotations 0, 2, 3, 4, 20200806)が含まれている。  
    jpeg形式で画像をもらっているが、モデルの学習・テストにはpng形式を用いている。  
    jpeg形式は圧縮方式なので、モデルの画素が安定しないため画像認識には向かないことを覚えておいて欲しい。  
    - train_detection_data, train_refined_images  
    検出モデルを学習するためのデータ。  
    detection_dataはモデルで読み込むためのアノテーションデータで、  
    refined_imagesは画像データである。  
    make_detection_data.ipynbを使って作成できる。  
    詳しくはdataset.mdに記載している。  
    - test_detection_data, test_refined_images  
    検出モデルをテストするためのデータ。  
    make_detection_data.ipynbを使って作成できる。  
    詳しくはdataset.mdに記載している。  
- dataset  
    - classification  
        - dataset.py  
        分類モデルのデータセットクラス。  
        - loader.py  
        分類モデルのデータ読み込み用関数をまとめたもの。  
        - sampler.py  
        データサンプリング用関数。  
        オーバーサンプリング、アンダーサンプリングは実装している。  
    - detection  
        - dataset.py  
        検出モデルのデータセットクラス。  
- evaluation  
    - classification  
        - evaluate.py  
        正答率と混同行列を計算する関数が入っている。  
        - statistics.py  
        pandasのデータフレームを作成する関数が入っている。  
        - visualize.py  
        可視化用関数がまとめてある。  
    - det2cls  
        - visualize.py  
        検出予測を可視化するための関数が入っている。  
        モデル予測と正解ラベルが同じ色で可視化されるようになっている。  
    - detection  
        - evaluate.py  
        検出モデルのVOC-APを計算するためのクラスが入っている。  
        Object_Detection_Metricsというライブラリを改造して作った。  
        - statistics.py  
        pandasのデータフレームを作成する関数が入っている。  
    - Object_Detection_Metrics  
    VOC-APを計算するライブラリ。  
    ところどころいじっている。  
- figure  
モデル予測や可視化した図などはここにいれている。  
- finished  
今は使わなくなったものを、ここに残してある。  
vis_classification_dataやvis_detection_dataは少し直せば使える可視化関数。  
他は説明しないが、研究の役に立つかもしれない。  
- IO  
    - build_ds.py  
    検出・分類データセットを構築する関数が入っている。  
    - create_bbox2size_ds.py  
    現在は使っていない。  
    昆虫の体サイズを計算する関数は入っており、  
    体サイズ関連の処理を書くときに役立つかもしれない。  
    - loader.py  
    様々なデータ読み込み用の関数が入っている。  
    - logger.py  
    学習・テスト時の引数を保存するロガーが入っている。  
    - utils.py  
    xml出力のための関数などが入っている。  
    PascalVoc形式のアノテーションを出力できる。  
    - visdom.py  
    visdomで可視化を行うための関数が入っている。  
- model  
    - mobilenet  
    作ってみたけど、性能は良くならなかったmobilenetを構築するクラス  
    - refinedet  
    現在検出モデルとして使用しているモデル。  
    モデル構築用のクラスや関数がまとめてある。  
    詳しくはコードを見て欲しい。  
    - resnet  
    現在分類モデル・体サイズ予測モデルとして使用しているモデル。  
    モデル構築用のクラスや関数がまとめてある。  
    詳しくはコードを見て欲しい。  
    unusedの中にAutoAugment/FastAutoAugmentの結果を適用する関数を  
    まとめてあるが、詳しくはreadme.mdを見て欲しい。  
    - segnet, unet  
    自分の以前の研究で使っていたコードを残してある。  
    セグメンテーションのモデルだが、役立つときがあるかもしれない。  
    - conv_layer.py  
    Group_Normalization + Weight_Standardizationを実装するために、  
    Conv2dを改造したもの。  
    - optimizer.py  
    AdamW, RAdamが入っている。  
- output_model  
学習したモデルはここに保存される。  
- output_xml  
出力した予測結果のxmlはここに保存される。  
- result  
解析した結果はここにまとめられている。  
- utils  
クロップやアノテーションのための関数が入っているが、  
ほとんど使っていない。  
- dataset.md  
データセット関連の情報はここに記載している。  
ここに載っていない情報は自分で調べて欲しい。  
- test_Refinedet_ResNet.ipynb  
自分の提案した資源量予測モデルを全体を通して動かすためのコード。  
- test_RefineDet.ipynb  
検出モデルを単体でテストするためのコード。  
- train_classification.ipynb  
分類モデルを学習するためのコード。  
日本語でコメントをつけた。  
- train_detection.ipynb  
検出モデルを学習するためのコード。  
日本語でコメントをつけた。  
- train_image2size.ipynb  
体サイズ予測モデルを学習するためのコード。  
- visualize_resnet_feature.ipynb  
ResNetの特徴量を可視化するためのコード。  
日本語でコメントをつけた。  

#### 現在の進捗状況  
自分の研究の流れはreadme.mdに記載しているが、特に記載しておきたいことだけまとめる。  
・分類モデルの最良  
転移学習 + オーバーサンプリング + ランダムデータ拡張  
モデルの場所: /home/tanida/workspace/Insect_Phenology_Detector/output_model/classification/master_paper/resnet50/b20_lr1e-5/crossvalid_20200806_OS_All5to6  
・検出モデルの最良  
転移学習 + ランダムデータ拡張  
モデルの場所: 
/home/tanida/workspace/Insect_Phenology_Detector/output_model/detection/RefineDet/master_paper/crop_b2/tcb5_im512_freeze_All0to2  
・体サイズ予測モデルの最良  
転移学習 + オーバーサンプリング + ランダムデータ拡張  
モデルの場所: 
/home/tanida/workspace/Insect_Phenology_Detector/output_model/image2size/ResNet34_b80_lr1e-4_OS_all02  

いろいろな手法を試したが、不均衡データの影響が大きすぎて転移学習・データ拡張などの学習量を正規化する  
手法しか良い効果を得られなかった。  
そのため、データそのものを増やすためにアノテーション作成の補助ツールを作る研究などが必要だと考えている。  
そして、モデルと人がうまく協力してアノテーションの精度・モデルの精度を高め合う仕組みが必要だと思う。  
機械学習のアプローチからはあまり良い結果が得られなかったが、生物学の分野から研究を進めると  
より良い機械学習手法が生み出せると自分は考えている。  

#### 研究の始め方  
各モデルの学習コードを動かしてみて、現在の処理の流れを確認して欲しい。  
また、テストコードを動かして現在のモデルの精度を確認して欲しい。  
ここまで動かせば処理の流れがつかめると思う。  

他にもデータセット作成用のコードなども用意してあるので、確認しながら研究を進めて欲しい。  