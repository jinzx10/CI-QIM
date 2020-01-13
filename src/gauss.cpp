#include <gauss.h>

using namespace arma;

double gauss(double const& x, double const& mu, double const& sigma) {
	return 1.0 / sigma / sqrt(2.0 * datum::pi) * 
		exp( -(x-mu)*(x-mu) / 2.0 / sigma / sigma );
}

mat gauss(vec const& x, rowvec const& y, double const& sigma) {
	mat z = zeros(x.n_elem, y.n_elem);
	for (uword j = 0; j != y.n_elem; ++j)
		z.col(j) = x - y(j);
	return exp( -0.5 * square(z/sigma) ) / ( sigma * sqrt(2.0 * datum::pi) );
}
