#include <iostream>
#include <armadillo>
#include <type_traits>
#include "template_helper.h"
#include "arma_helper.h"
#include "newtonroot.h"

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


int main() {

	auto f = [] (double x) { return x*x-3*x+2;};

	double x0 = 0;
	int status = newtonroot(f, x0);
	std::cout << status << std::endl << x0 << std::endl;

	x0 = 5;
	status = newtonroot(f, x0);
	std::cout  << status << std::endl<< x0 << std::endl;

	auto g  = [] (vec x) -> vec { return vec{x(0)-2*x(1), x(1)*x(1)-x(1)-2};};
	vec y0 = {0,0};
	status = newtonroot(g, y0);
	std::cout << status << std::endl;
	y0.print();

	
	return 0;
}
