#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include "Eigen/Dense"

class KalmanFilter
{
public:

	// state vector
	Eigen::VectorXd x_;

	// state covariance matrix
	Eigen::MatrixXd P_;

	// state transistion matrix
	Eigen::MatrixXd F_;

	// process covariance matrix
	Eigen::MatrixXd Q_;

	// identity matrix with the size equal to matrix P
	Eigen::MatrixXd I_;

	KalmanFilter() {}
	virtual ~KalmanFilter() {}

	/**
	* Init Initializes Kalman filter
	* @param x_in Initial state
	* @param P_in Initial state covariance
	* @param F_in Transition matrix
	* @param H_in Measurement matrix
	* @param R_in Measurement covariance matrix
	* @param Q_in Process covariance matrix
	*/
	void Init(Eigen::VectorXd &x_in, Eigen::MatrixXd &P_in, Eigen::MatrixXd &F_in, Eigen::MatrixXd &Q_in);

	/**
	* Prediction Predicts the state and the state covariance
	* using the process model
	* @param delta_T Time between k and k+1 in s
	*/
	void Predict();

	/**
	* Updates the state by using standard Kalman Filter equations
	* @param z The measurement at k+1
	*/
	void Update(const Eigen::VectorXd &z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R);

	/**
	* Updates the state by using Extended Kalman Filter equations
	* @param z The measurement at k+1
	*/
	void UpdateEKF(const Eigen::VectorXd& z, const Eigen::VectorXd& z_pred, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R);

private:
	inline void UpdateWithPredectedMeasurementDiff(const Eigen::VectorXd& y, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R);
};


inline void KalmanFilter::UpdateWithPredectedMeasurementDiff(const Eigen::VectorXd& y, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R)
{
	// Update the state by using Kalman Filter equations.
	// y is (z - z_pred).

	using Eigen::VectorXd;
	using Eigen::MatrixXd;

	MatrixXd Ht = H.transpose();
	MatrixXd PHt = P_ * Ht;
	MatrixXd S = H * PHt + R;
	MatrixXd K = PHt * S.inverse();

	// new estimate
	x_ = x_ + (K * y);
	P_ = (I_ - K * H) * P_;
}

#endif /* KALMAN_FILTER_H_ */
