#ifndef __AUXILIARY_MATH_H__
#define __AUXILIARY_MATH_H__

#include <armadillo>
#include <arma_helper.h>

// Fermi function
inline double fermi(double const& E, double const& mu, double const& kT) {
	return ( std::abs(kT) < arma::datum::eps ) ? 
		(E < mu) : 1.0 / ( std::exp( (E - mu) / kT ) + 1.0 );
}

inline arma::vec fermi(arma::vec const& E, double const& mu, double const& kT) {
	return ( std::abs(kT) < arma::datum::eps ) ? 
		arma::conv_to<arma::vec>::from(E < mu) : 1.0 / ( exp( (E - mu) / kT ) + 1.0 );
}

// Gaussian
inline double gauss(double const& x, double const& mu, double const& sigma) {
	return 1.0 / sigma / sqrt( 2.0 * arma::datum::pi ) 
		* exp( -(x-mu)*(x-mu) / 2.0 / sigma / sigma );
}

inline arma::mat gauss(arma::vec const& x, arma::rowvec const& y, double const& sigma) {
	return exp( -0.5*arma::square( bcast_op(x, y, std::minus<>()) / sigma ) ) 
		/ ( sigma * sqrt(2.0 * arma::datum::pi) );
}

// find the smallest number
inline double min(double const& i) {
	return i;
}

template <typename ...Ts>
double min(double const& i, Ts const& ...args) {
    double tmp = min(args...);
    return ( i < tmp ) ? i : tmp;
}

#endif
