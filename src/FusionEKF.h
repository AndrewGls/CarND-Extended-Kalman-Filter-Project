#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "kalman_filter.h"
#include "tools.h"

class FusionEKF
{
public:
	FusionEKF();
	virtual ~FusionEKF() {}

	// Run the whole flow of the Kalman Filter from here.
	void ProcessMeasurement(const MeasurementPackage &measurement_pack);

	// Kalman Filter update and prediction math lives in here.
	KalmanFilter ekf_;

private:
	void Init(const MeasurementPackage& pack);
	void Predict(const MeasurementPackage& pack);
	void Update(const MeasurementPackage& pack);
	Eigen::VectorXd PredictRadarMeasurement(const Eigen::VectorXd& x_state) const;

private:
	// check whether the tracking toolbox was initiallized or not (first measurement)
	bool is_initialized_;

	// previous timestamp
	long long previous_timestamp_;

	// tool object used to compute Jacobian and RMSE
	Eigen::MatrixXd R_laser_;
	Eigen::MatrixXd R_radar_;
	Eigen::MatrixXd H_laser_;
	Eigen::MatrixXd Hj_;
};

#endif /* FusionEKF_H_ */
