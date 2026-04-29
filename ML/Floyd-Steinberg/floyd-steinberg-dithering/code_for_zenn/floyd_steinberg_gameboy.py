import cv2
import numpy as np

# ゲームボーイ風のカラーパレット (RGB)
GAMEBOY_PALETTE = np.array([
    [15, 56, 15],     # 最暗色
    [48, 98, 48],     # 暗色
    [139, 172, 15],   # 明色
    [155, 188, 15]    # 最明色
], dtype=np.float64)

def find_closest_palette_color(pixel, palette):
    """RGB空間におけるユークリッド距離で一番近いパレット色を探す関数"""
    min_dist = float('inf')
    closest_color = palette[0]
    
    # 全パレット色との距離を計算
    for i in range(palette.shape[0]):
        # 距離の2乗（本来は平方根√を取りますが、大小比較だけなので計算が重い√は省略）
        dist = (pixel[0] - palette[i, 0])**2 + \
               (pixel[1] - palette[i, 1])**2 + \
               (pixel[2] - palette[i, 2])**2
        if dist < min_dist:
            min_dist = dist
            closest_color = palette[i]
            
    return closest_color
# 先ほどの GAMEBOY_PALETTE はそのまま使用

def floyd_steinberg_gray(img_float):
    """1次元（グレースケール）用の4段階ディザリング処理"""
    h, w = img_float.shape
    
    for y in range(h):
        for x in range(w):
            old_pixel = img_float[y, x]
            
            # RGBの距離計算をやめ、明るさを4段階(0, 85, 170, 255)に量子化する
            if old_pixel < 42.5: new_pixel = 0.0
            elif old_pixel < 127.5: new_pixel = 85.0
            elif old_pixel < 212.5: new_pixel = 170.0
            else: new_pixel = 255.0
                
            img_float[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            # 誤差拡散（チャンネルが1つになったので処理がシンプルに）
            if x + 1 < w: img_float[y, x + 1] += quant_error * (7.0 / 16.0)
            if y + 1 < h: img_float[y + 1, x] += quant_error * (5.0 / 16.0)
            if y + 1 < h and x - 1 >= 0: img_float[y + 1, x - 1] += quant_error * (3.0 / 16.0)
            if y + 1 < h and x + 1 < w: img_float[y + 1, x + 1] += quant_error * (1.0 / 16.0)
                
    return img_float

def floyd_steinberg_palette(img_float, palette):
    """任意のカラーパレットへのディザリング処理（純粋なPython実装）"""
    h, w, c = img_float.shape
    
    for y in range(h):
        for x in range(w):
            # 元のピクセル値を取得（参照渡しを防ぐためにcopy）
            old_pixel = np.copy(img_float[y, x])
            
            # 最も近いパレット色を取得
            new_pixel = find_closest_palette_color(old_pixel, palette)
            img_float[y, x] = new_pixel
            
            # 誤差ベクトル（R, G, Bそれぞれの誤差）を計算
            quant_error = old_pixel - new_pixel
            
            # 誤差を周囲に拡散（カラーなので3チャンネル同時に足し込む）
            if x + 1 < w:
                img_float[y, x + 1] += quant_error * (7.0 / 16.0)
            if y + 1 < h:
                img_float[y + 1, x] += quant_error * (5.0 / 16.0)
            if y + 1 < h and x - 1 >= 0:
                img_float[y + 1, x - 1] += quant_error * (3.0 / 16.0)
            if y + 1 < h and x + 1 < w:
                img_float[y + 1, x + 1] += quant_error * (1.0 / 16.0)
                
    return img_float

# --- 実行と計測 ---
# --- 実行部分の変更 ---

# 1. 画像をカラーではなく「グレースケール」で読み込む
img_gray = cv2.imread("input/winter-hitotsubashi.jpg", cv2.IMREAD_GRAYSCALE)
img_float = img_gray.astype(np.float64)

# 2. 1次元でのディザリングを実行
img_dithered_float = floyd_steinberg_gray(img_float)
img_dithered = np.clip(img_dithered_float, 0, 255).astype(np.uint8)

# 3. 出来上がった4段階の白黒画像に、ゲームボーイのパレットで「塗り絵」をする
h, w = img_dithered.shape
img_gameboy = np.zeros((h, w, 3), dtype=np.uint8)

# NumPyのブールインデックス参照で一括置換（ループを回すより圧倒的に速い！）
img_gameboy[img_dithered < 42] = GAMEBOY_PALETTE[0]
img_gameboy[(img_dithered >= 42) & (img_dithered < 127)] = GAMEBOY_PALETTE[1]
img_gameboy[(img_dithered >= 127) & (img_dithered < 212)] = GAMEBOY_PALETTE[2]
img_gameboy[img_dithered >= 212] = GAMEBOY_PALETTE[3]

# OpenCVの保存用にRGBからBGRに変換して保存
img_result_bgr = cv2.cvtColor(img_gameboy, cv2.COLOR_RGB2BGR)
cv2.imwrite("output/winter-hitotsubashi-gameboy-fixed.png", img_result_bgr)