from floyd_steinberg import floyd_steinberg_dither, floyd_steinberg_dither_njit
from floyd_steinberg_cpp import floyd_steinberg_dither_cpp
import time
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).parent

if __name__ == "__main__":
   img_gray = cv2.imread(str(BASE_DIR / "../input/winter-hitotsubashi.jpg"), cv2.IMREAD_GRAYSCALE)
   start_time = time.time()
   floyd_image = floyd_steinberg_dither(img_gray)
   end_time = time.time()
   print(f"通常変換時間: {end_time-start_time}") 
   start_time = time.time()
   floyd_image = floyd_steinberg_dither_njit(img_gray)
   end_time = time.time()
   print(f"Numba変換時間: {end_time-start_time}") 
   start_time = time.time()
   floyd_image = floyd_steinberg_dither_cpp(img_gray.tolist())
   end_time = time.time()
   print(f"C++変換時間: {end_time-start_time}") 
   