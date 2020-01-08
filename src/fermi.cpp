#include "../include/fermi.h"

double fermi(double const& E, double const& mu, double const& kT) {
	if ( std::abs(kT) < arma::datum::eps )
		return (E < mu);
	else
		return 1.0 / ( std::exp( (E - mu) / kT ) + 1.0 );
}

arma::vec fermi(arma::vec const& E, double const& mu, double const& kT) {
	if ( std::abs(kT) < arma::datum::eps)
		return arma::conv_to<arma::vec>::from(E < mu);
	else
		return 1.0 / ( arma::exp( (E - mu) / kT ) + 1.0 );
}
