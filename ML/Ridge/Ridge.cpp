#include <Eigen/Dense>

Eigen::VectorXd Ridge(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, double lambda) {
    auto n = X.cols();
    return (X.transpose() * X + lambda * Eigen::MatrixXd::Identity(n, n)).ldlt().solve(X.transpose() * y);
}

class StandardScaler {
    Eigen::RowVectorXd mean_, std_;
public:
    void fit(const Eigen::MatrixXd& X) {
        mean_ = X.colwise().mean();
        Eigen::MatrixXd c = X.rowwise() - mean_;
        auto var = c.array().square().colwise().sum() / (X.rows() - 1);
        std_ = var.sqrt().cwiseMax(1e-8).matrix();
    }

    Eigen::MatrixXd transform(const Eigen::MatrixXd& X) const {
        Eigen::MatrixXd out = X.rowwise() - mean_;
        for (int j = 0; j < out.cols(); ++j)
            out.col(j) /= std_(j);
        return out;
    }

    Eigen::MatrixXd fit_transform(const Eigen::MatrixXd& X) {
        fit(X);
        return transform(X);
    }
};

class RidgeRegressor {
    double lambda_;
    Eigen::VectorXd beta_;
public:
    explicit RidgeRegressor(double lambda = 1.0) : lambda_(lambda) {}

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        beta_ = Ridge(X, y, lambda_);
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const {
        return X * beta_;
    }

    Eigen::VectorXd get_beta() const { return beta_; }
};
