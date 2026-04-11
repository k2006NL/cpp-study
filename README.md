# C++ 学習記録

AtCoderをきっかけにC++を始めました(レート191 2026/04/11現在)。競技プログラミングで速いコードを書く面白さに触れるうちに、またKaggleでONNXを知る必要性が生じたことから「プロダクトの推論処理を高速化する」という実用的な方向にも興味が広がり、機械学習の実装も始めました。



## ディレクトリ構成

```
cpp-study/
├── ML/                        # 機械学習の実装
│   ├── Ridge/                 # Ridge回帰（Eigenによるコレスキー分解）
│   ├── ONNX/                  # ONNXモデルの推論高速化
│   └── bindings/              # pybind11によるPythonバインディング
├── ac-library-private/        # AtCoder用アルゴリズム実装（submodule）
└── third_party/               # 外部ライブラリ（Eigen, ONNX Runtime）
```

## 使用ライブラリ

- [Eigen](https://eigen.tuxfamily.org/) — 線形代数ライブラリ
- [ONNX Runtime](https://onnxruntime.ai/) — 機械学習モデルの推論エンジン
- [pybind11](https://github.com/pybind/pybind11) — C++/Pythonバインディング

## AIの利用

Claude Code: 環境構築、ディレクトリ構造の最適化、記述やコードに矛盾がないかについてのサポートをしていただきました。
Gemini: C++による実装ネタを教えていただきました。

C++のコード、Markdownは著者本人によるものです。
