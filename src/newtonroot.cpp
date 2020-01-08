#include "newtonroot.h"

double newtonroot(std::function<double(double)> f, double const& x0, double const& dx, double const& tol, unsigned int const& max_step) {
	unsigned int counter = 0;
	double x = x0;
	double J = 0;

	while (counter < max_step) {
		J = ( f(x+dx) - f(x-dx) ) / 2.0 / dx;
		double fx = f(x);
		x -= fx / J;
		//std::cout << x << std::endl;
		if ( std::abs(f(x)) < tol )
			break;
		counter += 1;
	}

	if ( counter >= max_step ) {
		std::cout << "Newton-Raphson method fails to find the root." << std::endl;
		return -1;
	} else {
		return x;
	}
}

arma::vec newtonroot(std::function<arma::vec(arma::vec)> f, arma::vec const& x0, double const& dx, double const& tol, unsigned int const& max_step) {
	unsigned int counter = 0;
	arma::vec x = x0;
	arma::uword len_x = x.size();
	arma::uword len_f = f(x).size();
	arma::mat J = arma::zeros(len_f, len_x);

	while (counter < max_step) {
		J.zeros();
		for (arma::uword i = 0; i != len_x; ++i) {
			arma::vec dxi = arma::zeros(len_x);
			dxi(i) = dx;
			J.col(i) = ( f(x+dxi) - f(x-dxi) ) / (2.0*dx);
		}
		x -= arma::solve(J, f(x));
		if ( arma::norm(f(x)) < tol )
			break;
		counter += 1;
	}

	if ( counter >= max_step ) {
		std::cout << "Newton-Raphson method fails to find the root." << std::endl;
		return arma::vec{-1};
	} else {
		return x;
	}
}


