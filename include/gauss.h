#ifndef __GAUSSIAN_H__
#define __GAUSSIAN_H__

#include <armadillo>

double gauss(double const& x, double const& mu, double const& sigma);
arma::mat gauss(arma::vec const& x, arma::rowvec const& y, double const& sigma);

#endif
