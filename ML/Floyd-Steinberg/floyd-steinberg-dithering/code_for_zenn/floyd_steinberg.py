import cv2
import numpy as np
from numba import njit


def simple_binarization(img_channel):
    """単純な閾値(128)による二値化（nodiffusion）"""
    img_float = img_channel.astype(float)
    h, w = img_float.shape
    
    for y in range(h):
        for x in range(w):
            old_pixel = img_float[y, x]
            # 128以上なら白(255)、未満なら黒(0)
            img_float[y, x] = 255.0 if old_pixel >= 128.0 else 0.0
            # 誤差拡散は行わない
            
    return np.clip(img_float, 0, 255).astype(np.uint8)

def floyd_steinberg_dither(img_channel):
    """Floyd-Steinberg法による誤差拡散ディザリング"""
    img_float = img_channel.astype(float)
    h, w = img_float.shape
    
    for y in range(h):
        for x in range(w):
            old_pixel = img_float[y, x]
            new_pixel = 255.0 if old_pixel >= 128.0 else 0.0
            img_float[y, x] = new_pixel
            
            # 誤差の計算
            quant_error = old_pixel - new_pixel 
            
            # 誤差を周囲に拡散
            if x + 1 < w:
                img_float[y, x + 1] += quant_error * 7.0 / 16.0
            if y + 1 < h:
                img_float[y + 1, x] += quant_error * 5.0 / 16.0
            if y + 1 < h and x - 1 >= 0:
                img_float[y + 1, x - 1] += quant_error * 3.0 / 16.0
            if y + 1 < h and x + 1 < w:
                img_float[y + 1, x + 1] += quant_error * 1.0 / 16.0
                
    return np.clip(img_float, 0, 255).astype(np.uint8)

@njit
def floyd_steinberg_dither_njit(img_channel):
    """Floyd-Steinberg法による誤差拡散ディザリング"""
    img_float = img_channel.astype(np.float64)
    h, w = img_float.shape
    
    for y in range(h):
        for x in range(w):
            old_pixel = img_float[y, x]
            new_pixel = 255.0 if old_pixel >= 128.0 else 0.0
            img_float[y, x] = new_pixel
            
            # 誤差の計算
            quant_error = old_pixel - new_pixel 
            
            # 誤差を周囲に拡散
            if x + 1 < w:
                img_float[y, x + 1] += quant_error * 7.0 / 16.0
            if y + 1 < h:
                img_float[y + 1, x] += quant_error * 5.0 / 16.0
            if y + 1 < h and x - 1 >= 0:
                img_float[y + 1, x - 1] += quant_error * 3.0 / 16.0
            if y + 1 < h and x + 1 < w:
                img_float[y + 1, x + 1] += quant_error * 1.0 / 16.0
                
    return np.clip(img_float, 0, 255).astype(np.uint8)

if __name__ == "__main__":

    # ① 元画像（カラー）の読み込みと保存
    img_color = cv2.imread("input/winter-hitotsubashi.jpg", cv2.IMREAD_COLOR)
    cv2.imwrite("output/winter-hitotsubashi.png", img_color)

    # グレースケールで読み込み直す
    img_gray = cv2.imread("input/winter-hitotsubashi.jpg", cv2.IMREAD_GRAYSCALE)

    # ② 単純な二値化（nodiffusion）の実行と保存
    img_gray_nodiffusion = simple_binarization(img_gray)
    cv2.imwrite("output/winter-hitotsubashi-nodiffusion.png", img_gray_nodiffusion)

    # ③ Floyd-Steinbergディザリングの実行と保存
    img_gray_floyd = floyd_steinberg_dither(img_gray)
    cv2.imwrite("output/winter-hitotsubashi-floyd.png", img_gray_floyd)

    # 3色を分解
    b, g, r = cv2.split(img_color)

    # それぞれに適用
    b_dither = floyd_steinberg_dither(b)
    g_dither = floyd_steinberg_dither(g)
    r_dither = floyd_steinberg_dither(r)

    # 結合
    img_color_dither = cv2.merge([b_dither, g_dither, r_dither])
    cv2.imwrite("output/winter-hitotsubashi-floyd-color.png", img_color_dither)
    print("画像の生成と保存が完了しました！")