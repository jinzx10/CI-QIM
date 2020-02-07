#include <iostream>
#include <armadillo>
#include "interp.h"

using namespace arma;

int main() {

	vec x = {-3, -1, 0, 0.5, 0.8, 0.9, 1};
	vec y = exp(-x%x/2);

	std::cout << lininterp(-4, x, y) << std::endl;;
	std::cout << lininterp(-3, x, y) << std::endl;;
	std::cout << lininterp(-2, x, y) << std::endl;;
	std::cout << lininterp(-1, x, y) << std::endl;;
	std::cout << lininterp(-0.5, x, y) << std::endl;;

	return 0;
}
