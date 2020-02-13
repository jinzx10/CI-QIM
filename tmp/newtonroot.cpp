#include <newtonroot.h>

using namespace arma;

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

	return ( counter >= max_step ) ? -1 : 0;
}

int newtonroot(std::function<vec(vec)> f, vec& x, double const& dx, double const& tol, unsigned int const& max_step) {
	unsigned int counter = 0;
	vec fx = f(x);
	uword len_x = x.n_elem;
	vec dxi = zeros(len_x);
	mat J = zeros(fx.n_elem, len_x);

	while (counter < max_step) {
		fx = f(x);
		if ( norm(fx) < tol )
			break;
		for (uword i = 0; i != len_x; ++i) {
			dxi.zeros();
			dxi(i) = dx;
			J.col(i) = ( f(x+dxi) - fx ) / dx;
		}
		x -= solve(J, fx);
		counter += 1;
	}

	return ( counter >= max_step ) ? -1 : 0;
}


