#include <iostream>
#include <Eigen/Dense>

Eigen::MatrixXd Ridge(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const double& lambda) {
  int col_num = X.cols();
  return (X.transpose() * X + lambda * Eigen::MatrixXd::Identity(col_num, col_num).inverse()) * X.transpose() * y;

}

