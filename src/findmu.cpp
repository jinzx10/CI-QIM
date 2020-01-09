#include "../include/findmu.h"
#include "../include/newtonroot.h"
#include "../include/fermi.h"

double findmu(arma::vec const& E, arma::uword const& n, double const& kT) {
	arma::vec val = arma::sort(E);
	if ( std::abs(kT) < arma::datum::eps )
		return val(n-1);

	auto dn = [&] (double const& mu) { return arma::accu(fermi(val, mu, kT)) - n; };
	double mu = val(n-1);
	newtonroot(dn, mu);
	return mu;
}
