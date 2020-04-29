#ifndef __MATH_HELPER_H__
#define __MATH_HELPER_H__

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
	return exp( -0.5*arma::square( bcast_minus(x, y) / sigma ) ) 
		/ ( sigma * sqrt(2.0 * arma::datum::pi) );
}


// find the smallest/largest number
inline double min(double const& i) {
	return i;
}

template <typename ...Ts>
double min(double const& i, Ts const& ...args) {
    double tmp = min(args...);
    return ( i < tmp ) ? i : tmp;
}

inline double max(double const& i) {
	return i;
}

template <typename ...Ts>
double max(double const& i, Ts const& ...args) {
    double tmp = max(args...);
    return ( i > tmp ) ? i : tmp;
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


// Newton-Raphson root-finding algorithm
// Broyden method TBD...
inline int newtonroot(std::function<double(double)> f, double& x, double const& dx = 1e-6, double const& tol = 1e-12, unsigned int const& max_step = 50) {
	unsigned int counter = 0;
	double fx = 0;
	double J = 0;

	while (counter < max_step) {
		fx = f(x);
		if ( std::abs(fx) < tol )
			break;
		J = ( f(x+dx) - fx ) / dx;
		x -= fx / J;
		counter += 1;
	}

	return ( counter >= max_step ) ? -1 : 0;
}

inline int newtonroot(std::function<arma::vec(arma::vec)> f, arma::vec& x, double const& dx = 1e-6, double const& tol = 1e-12, unsigned int const& max_step = 50) {
	unsigned int counter = 0;
	arma::vec fx = f(x);
	arma::uword len_x = x.n_elem;
	arma::vec dxi = arma::zeros(len_x);
	arma::mat J = arma::zeros(fx.n_elem, len_x);

	while (counter < max_step) {
		fx = f(x);
		if ( norm(fx) < tol )
			break;
		for (arma::uword i = 0; i != len_x; ++i) {
			dxi.zeros();
			dxi(i) = dx;
			J.col(i) = ( f(x+dxi) - fx ) / dx;
		}
		x -= solve(J, fx);
		counter += 1;
	}

	return ( counter >= max_step ) ? -1 : 0;
}

// solve for the chemical potential given the particle number and temperature
inline double findmu(arma::vec const& E, arma::uword const& n, double const& kT = 0.0) {
	arma::vec val = sort(E);
	if ( std::abs(kT) < arma::datum::eps )
		return val(n-1);

	auto dn = [&] (double const& mu) { return accu(fermi(val, mu, kT)) - n; };
	double mu = val(n-1);
	newtonroot(dn, mu);
	return mu;
}


// linear interpolation (or extrapolation, if outside the range)
inline double lininterp(double const& x0, arma::vec const& x, arma::vec const& y, bool is_evenly_spaced = false) {
	arma::uword i = 1;

	// x must be sorted in ascending order
	if ( is_evenly_spaced ) {
		if ( x0 > x(0) && x0 < x(x.n_elem-1) ) {
			i = (x0 - x(0)) / (x(1) - x(0)) + 1; 
		} else {
			if ( x0 >= x(x.n_elem-1) )
				i = x.n_elem - 1;
		}
	} else {
		// must not have repeated elements
		for (i = 1; i != x.n_elem-1; ++i)
			if ( x(i) > x0 ) break;

	}

	return y(i-1) + ( y(i) - y(i-1) ) / ( x(i) - x(i-1) ) * (x0 - x(i-1));
}

inline arma::rowvec lininterp(double const& x0, arma::vec const& x, arma::mat const& y, bool is_evenly_spaced = false) {
	arma::uword i = 1;

	// x must be sorted in ascending order
	if ( is_evenly_spaced ) {
		if ( x0 > x(0) && x0 < x(x.n_elem-1) ) {
			i = (x0 - x(0)) / (x(1) - x(0)) + 1; 
		} else {
			if ( x0 >= x(x.n_elem-1) )
				i = x.n_elem - 1;
		}
	} else {
		// must not have repeated elements
		for (i = 1; i != x.n_elem-1; ++i)
			if ( x(i) > x0 )
				break;
	}

	return y.row(i-1) + 
		( y.row(i) - y.row(i-1) ) / ( x(i) - x(i-1) ) * (x0 - x(i-1));
}


#endif
