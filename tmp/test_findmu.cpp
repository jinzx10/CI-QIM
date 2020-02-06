#include "../include/findmu.h"
#include "../include/fermi.h"

int main() {
	int sz = 10;
	arma::vec val = arma::randu(sz);
	double kT = 0.1;
	int n = 4;
	double mu = findmu(val, n, kT);
	std::cout << "mu = " << mu << std::endl;
	std::cout << "dn = " << std::abs(arma::accu(fermi(val, mu, kT))-n) << std::endl;
	return 0;
}
