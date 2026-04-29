#include <iostream>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

static Ort::Session CreateSession(Ort::Env& env, const std::string& model_path, int num_threads) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(num_threads);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    std::wstring wpath(model_path.begin(), model_path.end()); 
    return Ort::Session(env, wpath.c_str(), opts);
}

class BirdInferencer {
private:
  Ort::Env env_;
  Ort::Session session_;
  Ort::MemoryInfo memory_info_;

  // 入出力名の管理
  std::string input_name_str_;
  std::string output_name_str_;
  const char* input_name_[1];
  const char* output_name_[1];

public:   
  // コンストラクタによる初期化
  BirdInferencer(const std:: string& model_path, int num_threads = 4)
    : env_(ORT_LOGGING_LEVEL_WARNING, "BirdCLEF_Inference"),
      session_(CreateSession(env_, model_path, num_threads)),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
  {
    // モデルから入出力名を取得してキャッシュ
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_str_ = session_.GetInputNameAllocated(0, allocator).get();
    output_name_str_ = session_.GetOutputNameAllocated(0, allocator).get();
    input_name_[0] = input_name_str_.c_str();
    output_name_[0] = output_name_str_.c_str();
  }
  Eigen::VectorXd predict(const Eigen::MatrixXd& input_matrix) {
    // 1. スペクトログラムのサイズを取得
    int64_t H = input_matrix.rows(); // 周波数
    int64_t W = input_matrix.cols(); // 時間フレーム
    

    // 2. データの変換 （ONNXは行優先データを期待）
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> 
      row_major_data = input_matrix.cast<float>();


    // 3. テンソルの形状の定義 (Batch=1, Channel=3, H, W)
    std::vector<int64_t> input_shape = {1, 3, H, W};
    std::vector<float> input_tensor_values;
    input_tensor_values.reserve(3 * H * W);
    for (int i=0; i<3; i++) {
      input_tensor_values.insert(input_tensor_values.end(),
                                 row_major_data.data(),
                                 row_major_data.data() + row_major_data.size()
                                );
    }

    // 4. ONNXテンソルの作成
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info_,
      input_tensor_values.data(),
      input_tensor_values.size(),
      input_shape.data(),
      input_shape.size()
    );
    
    // 5. 推論実行
    auto output_tensors = session_.Run(
      Ort::RunOptions{nullptr},
      input_name_,
      &input_tensor,
      1,
      output_name_,
      1
    );

    // 6. 結果の取り出し
    float* float_arr = output_tensors.front().GetTensorMutableData<float>();
    auto shape_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    int64_t output_size = shape_info.GetElementCount();
    return Eigen::Map<Eigen::VectorXf>(float_arr, output_size).cast<double>();
  }

};

PYBIND11_MODULE(BirdEngine, m) {
    py::class_<BirdInferencer>(m, "BirdInferencer")
        .def(py::init<const std::string&, int>(), 
             py::arg("model_path"), 
             py::arg("num_threads") = 4)
        .def("predict", &BirdInferencer::predict, "スペクトログラム行列から鳥の種類を予測します");
}
