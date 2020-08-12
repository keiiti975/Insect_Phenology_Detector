### 現在保有しているデータ  

- all_classification_data  
    - classify_insect_std: 通常のデータセット  
    元のアノテーション: ["annotations_0", "annotations_2", "annotations_3", "annotations_4"]  
    ラベルマップ: {'Diptera': 0, 'Ephemeridae': 1, 'Ephemeroptera': 2, 'Lepidoptera': 3, 'Plecoptera': 4, 'Trichoptera': 5}  
    データ数: [408, 51, 178, 267, 130, 248]  
    - classify_insect_std_resizeFAR: アスペクト比を固定してリサイズしたデータセット  
    元のアノテーション: ["annotations_0", "annotations_2", "annotations_3", "annotations_4"]  
    ラベルマップ: {'Diptera': 0, 'Ephemeridae': 1, 'Ephemeroptera': 2, 'Lepidoptera': 3, 'Plecoptera': 4, 'Trichoptera': 5}  
    データ数: [408, 51, 178, 267, 130, 248]  
    - classify_insect_std_resize: アスペクト比を固定せずリサイズしたデータセット  
    元のアノテーション: ["annotations_0", "annotations_2", "annotations_3", "annotations_4"]  
    ラベルマップ: {'Diptera': 0, 'Ephemeridae': 1, 'Ephemeroptera': 2, 'Lepidoptera': 3, 'Plecoptera': 4, 'Trichoptera': 5}  
    データ数: [408, 51, 178, 267, 130, 248]  
    - classify_insect_std_plus_other: 検出対象クラス+その他のクラスを分類するデータセット  
    元のアノテーション: ["annotations_0", "annotations_2", "annotations_3", "annotations_4"]  
    ラベルマップ: {'Diptera': 0, 'Ephemeridae': 1, 'Ephemeroptera': 2, 'Lepidoptera': 3, 'Plecoptera': 4, 'Trichoptera': 5, 'Other': 6}  
    データ数: [408, 51, 178, 267, 130, 248, 2238]  
    - classify_insect_std_20200806: 通常のデータセット、2020/08/06にもらったデータまで  
    元のアノテーション: ["annotations_0", "annotations_2", "annotations_3", "annotations_4", "annotations_20200806"]  
    ラベルマップ: {'Diptera': 0, 'Ephemeridae': 1, 'Ephemeroptera': 2, 'Lepidoptera': 3, 'Plecoptera': 4, 'Trichoptera': 5}  
    データ数: [505, 143, 293, 1214, 493, 407]  
- train_detection_data  
    - refinedet_all: 全クラスをinsectラベルに集約したデータセット  
    元のアノテーション: ["annotations_0", "annotations_2", "annotations_3"]  
    ラベルマップ: {'insect': 0}  
    - refinedet_each: 個々のラベルを用いたデータセット  
    元のアノテーション: ["annotations_0", "annotations_2", "annotations_3"]  
    ラベルマップ: {'Coleoptera': 0, 'Diptera': 1, 'Ephemeridae': 2, 'Ephemeroptera': 3, 'Hemiptera': 4, 'Lepidoptera': 5, 'Plecoptera': 6, 'Trichoptera': 7, 'medium insect': 8, 'small insect': 9, 'snail': 10, 'spider': 11}  
    (存在しないクラスもあるかもしれない)  
    - refinedet_plus_other: 検出対象クラスを0、その他のクラスを1としたデータセット  
    元のアノテーション: ["annotations_0", "annotations_2", "annotations_3"]  
    ラベルマップ: {'insect': 0, 'other': 1}  
    - target_with_other: 評価用データセット、検出対象クラスに各ラベル、その他のクラスを6としたデータセット  
    元のアノテーション: ["annotations_0", "annotations_2", "annotations_3"]  
    ラベルマップ: {'Diptera': 0, 'Ephemeridae': 1, 'Ephemeroptera': 2, 'Lepidoptera': 3, 'Plecoptera': 4, 'Trichoptera': 5, 'Other': 6}  
    - refinedet_all_20200806: 全クラスをinsectラベルに集約したデータセット、2020/08/06にもらったデータまで  
    元のアノテーション: ["annotations_0", "annotations_2", "annotations_3", "annotations_20200806"]  
    ラベルマップ: {'insect': 0}  
    - refinedet_plus_other_20200806: 検出対象クラスを0、その他のクラスを1としたデータセット、2020/08/06にもらったデータまで  
    元のアノテーション: ["annotations_0", "annotations_2", "annotations_3", "annotations_20200806"]  
    ラベルマップ: {'insect': 0, 'other': 1}  
- test_detection_data  
    - refinedet_all: 全クラスをinsectラベルに集約したデータセット  
    元のアノテーション: ["annotations_4"]  
    ラベルマップ: {'insect': 0}  
    - refinedet_each: 個々のラベルを用いたデータセット  
    元のアノテーション: ["annotations_4"]  
    ラベルマップ: {'Coleoptera': 0, 'Diptera': 1, 'Ephemeridae': 2, 'Ephemeroptera': 3, 'Hemiptera': 4, 'Lepidoptera': 5, 'Plecoptera': 6, 'Trichoptera': 7, 'medium insect': 8, 'small insect': 9, 'snail': 10, 'spider': 11}  
    (存在しないクラスもあるかもしれない)  
    - refinedet_plus_other: 検出対象クラスを0、その他のクラスを1としたデータセット  
    元のアノテーション: ["annotations_4"]  
    ラベルマップ: {'insect': 0, 'other': 1}  
    - target_with_other: 評価用データセット、検出対象クラスに各ラベル、その他のクラスを6としたデータセット  
    元のアノテーション: ["annotations_4"]  
    ラベルマップ: {'Diptera': 0, 'Ephemeridae': 1, 'Ephemeroptera': 2, 'Lepidoptera': 3, 'Plecoptera': 4, 'Trichoptera': 5, 'Other': 6}  
    - refinedet_all_20200806: 全クラスをinsectラベルに集約したデータセット、2020/08/06にもらったデータまで  
    元のアノテーション: ["annotations_4"]  
    ラベルマップ: {'insect': 0}  
    - refinedet_plus_other_20200806: 検出対象クラスを0、その他のクラスを1としたデータセット、2020/08/06にもらったデータまで  
    元のアノテーション: ["annotations_4"]  
    ラベルマップ: {'insect': 0, 'other': 1}  