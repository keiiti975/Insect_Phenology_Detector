### 現在保有しているデータ  

- all_classification_data  
    - classify_insect_std: 通常のデータセット  
    元のアノテーション: ["annotations_0", "annotations_2", "annotations_3", "annotations_4"]  
    ラベルマップ: {'Coleoptera': 0, 'Diptera': 1, 'Ephemeridae': 2, 'Ephemeroptera': 3, 'Hemiptera': 4, 'Lepidoptera': 5, 'Plecoptera': 6, 'Trichoptera': 7}  
    データ数: [30, 419, 51, 200, 19, 271, 134, 250]  
    - classify_insect_std_resizeFAR: アスペクト比を固定してリサイズしたデータセット  
    元のアノテーション: ["annotations_0", "annotations_2", "annotations_3", "annotations_4"]  
    ラベルマップ: {'Coleoptera': 0, 'Diptera': 1, 'Ephemeridae': 2, 'Ephemeroptera': 3, 'Hemiptera': 4, 'Lepidoptera': 5, 'Plecoptera': 6, 'Trichoptera': 7}  
    データ数: [30, 419, 51, 200, 19, 271, 134, 250]  
    - classify_insect_std_resize: アスペクト比を固定せずリサイズしたデータセット  
    元のアノテーション: ["annotations_0", "annotations_2", "annotations_3", "annotations_4"]  
    ラベルマップ: {'Coleoptera': 0, 'Diptera': 1, 'Ephemeridae': 2, 'Ephemeroptera': 3, 'Hemiptera': 4, 'Lepidoptera': 5, 'Plecoptera': 6, 'Trichoptera': 7}  
    データ数: [30, 419, 51, 200, 19, 271, 134, 250]  
- train_detection_data  
    - refinedet_all: 全クラスをinsectラベルに集約したデータセット  
    元のアノテーション: ["annotations_0", "annotations_2", "annotations_3"]  
    ラベルマップ: {'insect': 0}  
    - refinedet_each: 個々のラベルを用いたデータセット  
    元のアノテーション: ["annotations_0", "annotations_2", "annotations_3"]  
    ラベルマップ: {'Coleoptera': 0, 'Diptera': 1, 'Ephemeridae': 2, 'Ephemeroptera': 3, 'Hemiptera': 4, 'Lepidoptera': 5, 'Plecoptera': 6, 'Trichoptera': 7, 'medium insect': 8, 'small insect': 9, 'snail': 10, 'spider': 11}  
    (存在しないクラスもあるかもしれない)  
- test_detection_data  
    - refinedet_all: 全クラスをinsectラベルに集約したデータセット  
    元のアノテーション: ["annotations_4"]  
    ラベルマップ: {'insect': 0}  
    - refinedet_each: 個々のラベルを用いたデータセット  
    元のアノテーション: ["annotations_4"]  
    ラベルマップ: {'Coleoptera': 0, 'Diptera': 1, 'Ephemeridae': 2, 'Ephemeroptera': 3, 'Hemiptera': 4, 'Lepidoptera': 5, 'Plecoptera': 6, 'Trichoptera': 7, 'medium insect': 8, 'small insect': 9, 'snail': 10, 'spider': 11}  
    (存在しないクラスもあるかもしれない)  