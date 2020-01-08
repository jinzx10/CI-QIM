#include "../include/fermi.h"

int main() {
	arma::uword sz = 10;
	arma::vec E = arma::randu(sz);
	double kT = 1e-1;
	double mu = 0.5;
	fermi(E,mu,kT).print();

	return 0;
}
