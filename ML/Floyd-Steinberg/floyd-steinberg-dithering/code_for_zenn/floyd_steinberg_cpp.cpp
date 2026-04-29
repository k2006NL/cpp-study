#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>

using std::vector;

vector<vector<int>> floyd_steinberg_dither_cpp(vector<vector<int>> img_vec) {
    int h = static_cast<int>(img_vec.size());
    int w = static_cast<int>(img_vec[0].size());
    vector<vector<double>> img_float(h, vector<double>(w));
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            img_float[y][x] = static_cast<double>(img_vec[y][x]);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double old_pixel = img_float[y][x];
            double new_pixel = (old_pixel >= 128) ? 255.0 : 0.0;
            double quant_error = old_pixel - new_pixel;

            if (x + 1 < w)             img_float[y][x+1]   += quant_error * 7.0 / 16.0;
            if (y + 1 < h)             img_float[y+1][x]   += quant_error * 5.0 / 16.0;
            if (y + 1 < h && x - 1 >= 0) img_float[y+1][x-1] += quant_error * 3.0 / 16.0;
            if (y + 1 < h && x + 1 < w)  img_float[y+1][x+1] += quant_error * 1.0 / 16.0;
        }
    }

    vector<vector<int>> img_floyd(h, vector<int>(w));
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            img_floyd[y][x] = static_cast<int>(std::clamp(img_float[y][x], 0.0, 255.0));
    return img_floyd;
}

PYBIND11_MODULE(floyd_steinberg_cpp, m) {
    m.def("floyd_steinberg_dither_cpp", &floyd_steinberg_dither_cpp);
}
