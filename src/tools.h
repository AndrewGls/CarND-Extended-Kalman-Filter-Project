#ifndef TOOLS_H_
#define TOOLS_H_

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits> 
#include "Eigen/Dense"

namespace Tools
{
	extern const double epsilon;

	// A helper method to calculate RMSE.
	Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, const std::vector<Eigen::VectorXd> &ground_truth);

	// A helper method to calculate Jacobians.
	Eigen::MatrixXd CalculateJacobian(const Eigen::VectorXd& x_state);

	// Returns projected values to X and Y as 2D vector.
	Eigen::VectorXd PolarToCartesian(double rho, double phi);
};


inline Eigen::VectorXd Tools::PolarToCartesian(double rho, double phi)
{
	Eigen::Vector2d result;
	result << rho * cos(phi), rho * sin(phi);
	return result;
}

inline Eigen::MatrixXd Tools::CalculateJacobian(const Eigen::VectorXd& x_state)
{
	Eigen::MatrixXd Hj(3, 4);
	//recover state parameters
	double px = x_state(0);
	double py = x_state(1);
	double vx = x_state(2);
	double vy = x_state(3);

	// pre-compute a set of terms to avoid repeated calculation
	// check division by zero is replaced by max().
	double c1 = px*px + py*py;
	c1 = std::max(c1, epsilon);
	double c2 = sqrt(c1);
	c2 = std::max(c2, epsilon);
	double c3 = (c1*c2);
	c3 = std::max(c3, epsilon);

	//compute the Jacobian matrix
	Hj << (px / c2), (py / c2), 0, 0,
		-(py / c1), (px / c1), 0, 0,
		py*(vx*py - vy*px) / c3, px*(px*vy - py*vx) / c3, px / c2, py / c2;

	return Hj;
}

#endif /* TOOLS_H_ */
