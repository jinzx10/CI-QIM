#include <interp.h>

using namespace arma;

int main() {
	vec x = linspace(-2, 3, 100);
	vec y = (x-1)%(x-1);

	std::cout << "f(1.1) = " << lininterp_linspace(1.1, x, y) << std::endl;
	std::cout << "f(-1.1) = " << lininterp_linspace(-1.1, x, y) << std::endl;
	std::cout << "f(-2.16) = " << lininterp_linspace(-2.16, x, y) << std::endl;
	std::cout << "f(3.23) = " << lininterp_linspace(3.23, x, y) << std::endl;

	return 0;
}
