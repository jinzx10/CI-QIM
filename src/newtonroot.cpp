#include "../include/newtonroot.h"

int newtonroot(std::function<double(double)> f, double& x, double const& dx, double const& tol, unsigned int const& max_step) {
	unsigned int counter = 0;
	double fx = 0;
	double J = 0;

	while (counter < max_step) {
		fx = f(x);
		if ( std::abs(fx) < tol )
			break;
		J = ( f(x+dx) - fx ) / dx;
		x -= fx / J;
		counter += 1;
	}

	if ( counter >= max_step ) {
		return -1;
	} else {
		return 0;
	}
}

int newtonroot(std::function<arma::vec(arma::vec)> f, arma::vec& x, double const& dx, double const& tol, unsigned int const& max_step) {
	unsigned int counter = 0;
	arma::vec fx = f(x);
	arma::uword len_x = x.size();
	arma::uword len_f = fx.size();
	arma::mat J = arma::zeros(len_f, len_x);

	while (counter < max_step) {
		fx = f(x);
		if ( arma::norm(fx) < tol )
			break;
		for (arma::uword i = 0; i != len_x; ++i) {
			arma::vec dxi = arma::zeros(len_x);
			dxi(i) = dx;
			J.col(i) = ( f(x+dxi) - f(x) ) / dx;
		}
		x -= arma::solve(J, fx);
		counter += 1;
	}

	if ( counter >= max_step ) {
		return -1;
	} else {
		return 0;
	}
}


