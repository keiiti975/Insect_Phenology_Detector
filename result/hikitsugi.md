### 引き継ぎ資料  

#### 各結果の場所について  
修士論文で用いたデータはmaster_paperに保存している  
classification: 分類モデルの結果  
detection: 検出モデルの結果  
resource_estimation: 資源量予測モデルの結果  
size_estimation: 体サイズ予測モデルの結果  

それまでに得られた結果はresult直下のフォルダに分けて保存している  
色々な結果があるけど、修士論文に書いたモデルが最も良くなったので参考にはならないと思う  

#### その他の重要なファイル  
以下に示すファイルは研究を行う上で重要です  
・result/classification/visualize_annotation  
annotation_0 ~ 4の解析結果  
・result/classification/visualize_annotation  
annotation_0 ~ 4 + annotation_20200806の解析結果  
・result/classification/dataset_distributions  
annotation_0 ~ 4 + annotation_20200806のデータを、クラス分布として可視化したもの  
・result/detection/visualize_bounding_box  
annotation_0 ~ 4 + annotation_20200806のデータを、クラス別のバウンディングボックスの大きさの観点から可視化したもの  
・result/ilsvrc2012_analysis  
ILSVRC2012データセットを分析して、クラスカテゴリの分布の可視化と昆虫の占める割合を調べたもの  

#### その他注意  
compare_* と書いてあるファイルはモデルを比較したものです  
visualize_* と書いてあるファイルは何かを可視化したものです  