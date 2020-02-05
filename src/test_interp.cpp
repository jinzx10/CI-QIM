#include <interp.h>

using namespace arma;

int main() {
	vec x = linspace(-2, 3, 100);
	vec y = (x-1)%(x-1);

	std::cout << "f(1.1) = " << lininterp_linspace(1.1, x, y) << std::endl;
	std::cout << "f(-1.1) = " << lininterp_linspace(-1.1, x, y) << std::endl;
	std::cout << "f(-2.16) = " << lininterp_linspace(-2.16, x, y) << std::endl;
	std::cout << "f(3.23) = " << lininterp_linspace(3.23, x, y) << std::endl;

	std::string datadir = "/home/zuxin/job/CI-QIM/data/test_TwoPara/";

	vec xgrid, E0;
	xgrid.load(datadir+"xgrid.txt", arma::raw_ascii);
	E0.load(datadir+"E0.txt", arma::raw_ascii);


	uword nx = 2000;
	vec x_fine = linspace(-20, 40, nx);
	vec E0_fine = zeros(nx);

	for (uword i = 0; i != nx; ++i) {
		double x = x_fine(i);
		E0_fine(i) = lininterp_linspace(x, xgrid, E0);
	}

	E0_fine.save("interp.txt", raw_ascii);


	return 0;
}
