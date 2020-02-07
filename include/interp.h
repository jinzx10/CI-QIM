#ifndef __INTERPOLATION_H__
#define __INTERPOLATION_H__

#include <armadillo>

double lininterp_linspace(double const& x0, arma::vec const& x, arma::vec const& y);
double lininterp(double const& x0, arma::vec const& x, arma::vec const& y);

#endif
