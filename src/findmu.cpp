#include <findmu.h>
#include <newtonroot.h>
#include <fermi.h>

using namespace arma;

double findmu(vec const& E, uword const& n, double const& kT) {
	vec val = sort(E);
	if ( std::abs(kT) < datum::eps )
		return val(n-1);

	auto dn = [&] (double const& mu) { return accu(fermi(val, mu, kT)) - n; };
	double mu = val(n-1);
	newtonroot(dn, mu);
	return mu;
}
