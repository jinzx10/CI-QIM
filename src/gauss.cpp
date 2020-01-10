#include "../include/gauss.h"

using namespace arma;

double gauss(double const& x, double const& mu, double const& sigma) {
	return 1.0 / sigma / sqrt(2.0 * datum::pi) * 
		std::exp( -(x-mu)*(x-mu) / 2.0 / sigma / sigma );
}

mat gauss(vec const& x, rowvec const& y, double const& sigma) {
	return 1.0 / sigma / sqrt(2.0 * datum::pi) * 
		exp( -square( repmat(x, 1, y.n_elem) - repmat(y, x. n_elem,1) ) / 2.0 / sigma / sigma );
}
