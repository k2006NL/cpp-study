import torch
import timm
import onnx

# 鳥コンペ(Bird CLEF+2026)に向けた学習という前提

# 1. timm(Pytorch Image Models) の読み込み
model_name = 'resnet18'
model = timm.create_model(model_name, pretrained=True, num_classes=234)
model.eval()

# 2. ダミー入力の生成
dummy_input = torch.randn(1, 3, 128, 1249) # それぞれ(バッチサイズ, RGB, 周波数, 時間フレーム数)に対応

onnx_file_path = f"{model_name}.onnx"

# 3. ONNX エクスポート
torch.onnx.export(
    model,
    dummy_input,
    onnx_file_path,
    export_params=True,           # 学習済重みを保存
    opset_version=17,
    do_constant_folding=True,     # 定数畳込み
    input_names=["input"],        # C++側で参照する名前    
    output_names=["output"],      # C++側で参照する名前
    
    # 動的軸の設定(入力の内、固定しない次元を指す)
    dynamic_axes={
        "input": {0: "batch_size", 3: "width"},
        "output": {0: "batch_size"}
    }
)

print(f"モデルを{onnx_file_path}に保存")

# 4. モデルの検証
onnx_model = onnx.load(onnx_file_path)
try:
    onnx.checker.check_model(onnx_model)
    print("モデル検証○")
except onnx.checker.ValidationError as e:
    print("モデル検証×: {e}")







