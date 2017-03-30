#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in, MatrixXd &Q_in)
{
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  Q_ = Q_in;
  I_ = MatrixXd::Identity(P_.rows(), P_.cols());
}

void KalmanFilter::Predict()
{
	x_ = F_ * x_;
	P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const Eigen::VectorXd &z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R)
{
   // Update the state by using Kalman Filter equations.

	VectorXd z_pred = H * x_;
	UpdateWithPredectedMeasurementDiff(z - z_pred, H, R);
}

void KalmanFilter::UpdateEKF(const Eigen::VectorXd& z, const Eigen::VectorXd& z_pred, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R)
{
	UpdateWithPredectedMeasurementDiff(z - z_pred, H, R);
}
