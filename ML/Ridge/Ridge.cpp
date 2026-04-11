#include <iostream>
#include <Eigen/Dense>


Eigen::MatrixXd Ridge(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const double& lambda) {
  int col_num = X.cols();
  // コレスキー分解による解法
  return (X.transpose() * X + lambda * Eigen::MatrixXd::Identity(col_num, col_num)).ldlt().solve(X.transpose() * y);
}

