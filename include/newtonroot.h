#ifndef __NEWTON_RAPHSON_H__
#define __NEWTON_RAPHSON_H__

#include <functional>
#include <armadillo>

double newtonroot(std::function<double(double)> f, double const& x0, double const& dx = 1e-6, double const& tol = 1e-12, unsigned int const& max_step = 50); 
arma::vec newtonroot(std::function<arma::vec(arma::vec)> f, arma::vec const& x0, double const& dx = 1e-6, double const& tol = 1e-12, unsigned int const& max_step = 50);

#endif
