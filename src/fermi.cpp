#include <fermi.h>

using namespace arma;

double fermi(double const& E, double const& mu, double const& kT) {
	return ( std::abs(kT) < datum::eps ) ?  (E < mu) : 1.0 / ( std::exp( (E - mu) / kT ) + 1.0 );
}

vec fermi(vec const& E, double const& mu, double const& kT) {
	return ( std::abs(kT) < datum::eps ) ?  conv_to<vec>::from(E < mu) : 1.0 / ( exp( (E - mu) / kT ) + 1.0 );
}
