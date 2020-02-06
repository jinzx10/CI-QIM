#include "../include/newtonroot.h"

double f1(double x) {
	return (x-2.2)*(x-3.3);
}

arma::vec f2(arma::vec x) {
	return arma::vec{x(0)-1.1, x(1)*x(1)-3*x(0)*x(0)  };
}

int main() {
	int status = 0;
	double x = 1;
	status = newtonroot(f1,x);
	std::cout << "status = " << status << std::endl
		<< "root of f1 = " << x << std::endl;
	x = 4;
	status = newtonroot(f1,x);
	std::cout << "status = " << status << std::endl
		<< "root of f1 = " << x << std::endl;

	arma::vec y = {1,1};
	status = newtonroot(f2,y);
	std::cout << "status = " << status << std::endl;
	y.print();

	return 0;
}
