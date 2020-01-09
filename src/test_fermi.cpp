#include "../include/fermi.h"

int main() {
	double kT = 5e-2;
	double mu = 0.7;
	std::cout << "kT = " << kT << "   mu = " << mu << std::endl;

	double E = arma::randu();
	std::cout << "E = " << E << " " << "f = " << fermi(E,mu,kT) << std::endl;

	arma::uword sz = 10;
	arma::mat fE = arma::zeros(sz,2);
	fE.col(0) = arma::randu(sz);
	fE.col(1) = fermi(fE.col(0),mu,kT);
	std::cout << "     E        f" << std::endl;
	fE.print();

	return 0;
}
