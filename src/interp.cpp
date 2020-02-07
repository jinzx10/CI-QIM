#include "interp.h"

using namespace arma;

double lininterp_linspace(double const& x0, vec const& x, vec const& y) {
	// x is sorted in ascending order and evenly-spaced
	double dx = x(1) - x(0);
	uword i = 0;

	// find i such that y0 is interpolated/extrapolated from x(i) and x(i+1)
	if ( x0 > x(0) && x0 < x(x.n_elem-1) ) { // if x0 is within the range of x
		i = (x0 - x(0)) / dx; 
	} else {
		if ( x0 >= x(x.n_elem-1) )
			i = x.n_elem-2;
	}
	double k = ( y(i+1) - y(i) ) / dx;
	return y(i) + k * (x0 - x(i));
}

double lininterp(double const& x0, vec const& x, vec const& y) {
	// x is sorted in ascending order
	// not necesarily evenly-spaced, but must not have repeated elements
	uword i;
	for (i = 1; i != x.n_elem-1; ++i)
		if ( x(i) > x0 ) break;

	double k = ( y(i) - y(i-1) ) / ( x(i) - x(i-1) );
	return y(i-1) + k * (x0 - x(i-1));
}
