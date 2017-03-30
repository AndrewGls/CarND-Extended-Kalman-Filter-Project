#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


namespace ns_init_state
{
	// values from project suggestion

	const double laser_var_px = 0.0225;
	const double laser_var_py = 0.0225;

	const double radar_var_rho = 0.09;
	const double radar_var_phi = 0.0009;
	const double radar_var_rho_dot = 0.09;

	const double var_pos = 1.;
	const double var_vel = 1000.;

	const double noise_ax = 9; 
	const double noise_ay = 9; 

	const double delta_t = 0.001; // time resolution for prediction state
}

FusionEKF::FusionEKF()
	: is_initialized_ (false)
	, previous_timestamp_ (0)
	, R_laser_ (MatrixXd(2, 2))
	, R_radar_ (MatrixXd(3, 3))
	, H_laser_ (MatrixXd(2, 4))
	, Hj_ (MatrixXd(3, 4))
{
	using namespace ns_init_state;

	// measurement covariance matrix - laser
	R_laser_ << laser_var_px, 0,
				0, laser_var_py;

	// measurement covariance matrix - radar
	R_radar_ << radar_var_rho, 0, 0,
				0, radar_var_phi, 0,
				0, 0, radar_var_rho_dot;

	H_laser_ << 1, 0, 0, 0,
				0, 1, 0, 0;

	Hj_ << 0, 0, 0, 0,
		   0, 0, 0, 0,
		   0, 0, 0, 0;
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage& pack)
{
	//  Initialization
	if (!is_initialized_) {
		Init(pack);
		return;
	}

	// Do prediction.
	// Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
	Predict(pack);
	
	// Do Update.
	Update(pack);

	// print the output
	//cout << "x_ = " << ekf_.x_ << endl;
	//cout << "P_ = " << ekf_.P_ << endl;
}


void FusionEKF::Init(const MeasurementPackage& pack)
{
	assert(!is_initialized_);

	// Initialize the state ekf_.x_ with the first measurement.
	//  Create the covariance matrix.
	//  Remember: you'll need to convert radar from polar to cartesian coordinates.

	using namespace ns_init_state;

	auto P = MatrixXd(4, 4);
	P << var_pos, 0, 0, 0,
		 0, var_pos, 0, 0,
		 0, 0, var_vel, 0,
		 0, 0, 0, var_vel;

	auto F = MatrixXd(4, 4);
	F << 1, 0, 1, 0,
		 0, 1, 0, 1,
		 0, 0, 1, 0,
		 0, 0, 0, 1;

	auto Q = MatrixXd(4, 4);
	Q << 0, 0, 0, 0,
		 0, 0, 0, 0,
		 0, 0, 0, 0,
		 0, 0, 0, 0;

	// first measurement

	auto x = VectorXd(4);

	previous_timestamp_ = pack.timestamp_;

	if (pack.sensor_type_ == MeasurementPackage::RADAR) {
		// Convert radar from polar to cartesian coordinates and initialize state as (px, py, vx, vy)
		const auto rho = pack.raw_measurements_(0, 0);
		const auto phi = pack.raw_measurements_(1, 0);
		const auto rho_dot = pack.raw_measurements_(2, 0);
		const auto pos = Tools::PolarToCartesian(rho, phi);			// returns (px, py)
		const auto vel = Tools::PolarToCartesian(rho_dot, phi);		// returns (vx, vy)
		x << pos(0, 0), pos(1, 0), vel(0, 0), vel(1, 0);			// sets (px, py, vx, vy)
	}
	else if (pack.sensor_type_ == MeasurementPackage::LASER) {
		// Initialize state as (px, py, vx, vy)
		x << pack.raw_measurements_[0], pack.raw_measurements_[1], 0, 0;
	}

	ekf_.Init(x, P, F, Q);

	// done initializing, no need to predict or update
	is_initialized_ = true;
}

void FusionEKF::Predict(const MeasurementPackage& pack)
{
	// Time is measured in seconds.
	const double dt = (pack.timestamp_ - previous_timestamp_) / 1.e6;
	assert(dt >= 0.);
	if (dt < ns_init_state::delta_t) {
		return; // skip recalculation for similar measured values.
	}

	previous_timestamp_ = pack.timestamp_;

	using namespace ns_init_state;

	// Update the State Transition Matrix according to the new elapsed time:
	ekf_.F_(0, 2) = dt;
	ekf_.F_(1, 3) = dt;

	// Update the Noise Process Covariance Matrix Q:
	const auto dt_2 = dt * dt;
	const auto dt_3 = dt_2 * dt;
	const auto dt_4 = dt_3 * dt;
	ekf_.Q_ << dt_4/4 * noise_ax, 0, dt_3/2 * noise_ax, 0,
			   0, dt_4/4 * noise_ay, 0, dt_3/2 * noise_ay,
			   dt_3/2 * noise_ax, 0, dt_2*noise_ax, 0,
			   0, dt_3/2 * noise_ay, 0, dt_2*noise_ay;

	// Do prediction.
	ekf_.Predict();
}

void FusionEKF::Update(const MeasurementPackage& pack)
{
	// Performs the update state for specified sensor type.
	// Updates the state and covariance matrices.

	if (pack.sensor_type_ == MeasurementPackage::RADAR) {
		// Radar updates
		Hj_ = Tools::CalculateJacobian(ekf_.x_);
		const auto z_pred = PredictRadarMeasurement(ekf_.x_);
		ekf_.UpdateEKF(pack.raw_measurements_, z_pred, Hj_, R_radar_);
	}
	else {
		// Laser updates
		ekf_.Update(pack.raw_measurements_, H_laser_, R_laser_);
	}
}

VectorXd FusionEKF::PredictRadarMeasurement(const VectorXd& x_state) const
{
	const auto px = x_state(0, 0);
	const auto py = x_state(1, 0);
	const auto vx = x_state(2, 0);
	const auto vy = x_state(3, 0);

	const auto rho = sqrt(px * px + py * py);
	const auto phi = atan2(py, px);
	const auto rho_dot = (px * vx + py * vy) / (rho + Tools::epsilon);

	VectorXd result(3);
	result << rho, phi, rho_dot;
	return result;
}