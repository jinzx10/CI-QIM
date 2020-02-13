#ifndef __AUXILIARY_MATH_H__
#define __AUXILIARY_MATH_H__

#include <armadillo>
#include <arma_helper.h>
#include <functional>


// Fermi function
inline double fermi(double const& E, double const& mu, double const& kT) {
	return ( std::abs(kT) < arma::datum::eps ) ? 
		(E < mu) : 1.0 / ( std::exp( (E - mu) / kT ) + 1.0 );
}

inline arma::vec fermi(arma::vec const& E, double const& mu, double const& kT) {
	return ( std::abs(kT) < arma::datum::eps ) ? 
		arma::conv_to<arma::vec>::from(E < mu) : 1.0 / ( exp( (E - mu) / kT ) + 1.0 );
}


// Boltzmann weight
inline arma::vec boltzmann(arma::vec const& E, double const& kT) {
	arma::uword imin = E.index_min();
	return ( std::abs(kT) < arma::datum::eps ) ?
		unit_vec(E.n_elem, imin) : arma::exp(-(E-E(imin))/kT) / arma::accu( arma::exp(-(E-E(imin))/kT) );
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


// numerical gradient of real functions by finite difference
inline std::function<double(double)> grad(std::function<double(double)> const& f, double const& delta = 0.001) {
	return [=] (double x) -> double {
		return ( -f(x-3.0*delta)/60.0 + 3.0*f(x-2.0*delta)/20.0 - 3.0*f(x-delta)/4.0 +
				f(x+3.0*delta)/60.0 - 3.0*f(x+2.0*delta)/20.0 + 3.0*f(x+delta)/4.0 ) 
			/ delta;
	};
}

template <typename V>
std::function<double(V)> gradi(std::function<double(V)> const& f, size_t const& i, double const& delta = 0.001) {
	return [=] (V const& v) -> double {
		std::function<double(double)> g = [=, v=v] (double const& x) mutable {
			v[i] = x;
			return f(v);
		};
		return grad(g, delta)(v[i]); 
	};
}

template <typename V>
std::function<V(V)> grad(std::function<double(V)> const& f, double const& delta = 0.001) {
	return [=] (V const& x) -> V {
		V df = x;
		for (size_t i = 0; i != x.size(); ++i) {
			df[i] = gradi(f, i, delta)(x);
		}
		return df;
	};
}


// generate grid points according to some grid density
inline arma::vec grid(double const& xmin, double const& xmax, std::function<double(double)> density) {
	arma::vec g = {xmin};
	double g_last = g.back();
	while ( g_last < xmax ) {
		g.resize(g.n_elem+1);
		g.back() = g_last + 1.0 / density(g_last);
		g_last = g.back();
	}
	return g;
}


#endif
