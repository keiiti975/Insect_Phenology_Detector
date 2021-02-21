### AutoAugment/FastAutoAugmentデータ拡張の使い方

1. AutoAugmentクラスを作成  
この時、policy_dirを与えれば昆虫の分類に適応したFastAutoAugmentの結果を使用できる。  
与えなければ、AutoAugmentで指摘された最良データ拡張を使用できる。  

2. AutoAugment(img)の形でデータ拡張を行える  

FastAutoAugmentのコードも残してあるので、tanidaのフォルダを探してもらえば使えるはずです。  
だけど、今のResNetのコードでは動かないと思うのでその部分は修正をお願いします。  