Day 1〜2 (5/3〜5/4): 基礎概念

量子化とは何か (FP32 → INT8/INT4)
なぜ精度が大きく落ちないのか (量子化誤差と分布の話)
動的量子化 vs 静的量子化 vs QAT (Quantization-Aware Training)
学習リソース: ONNX Runtime公式ドキュメント、Hugging Face の量子化ガイド

Day 3〜4 (5/5〜5/6): llama.cpp 周辺

GGUF形式とは
Q4_K_M, Q5_K_M, Q8_0 などの記号の意味
量子化レベル別の精度・速度・メモリのトレードオフ
学習リソース: llama.cpp の README、量子化解説ブログ複数

Day 5〜7 (5/7〜5/9): 実装

llama.cpp ビルド
既存GGUFモデルダウンロード (Hugging Face TheBloke、mmnga など)
推論実行、ベンチマーク取得
学習リソース: llama.cpp 公式の examples
