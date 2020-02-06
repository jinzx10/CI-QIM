#ifndef __NEWTON_RAPHSON_H__
#define __NEWTON_RAPHSON_H__

#include <functional>
#include <armadillo>

int newtonroot(std::function<double(double)> f, double& x, double const& dx = 1e-6, double const& tol = 1e-12, unsigned int const& max_step = 50); 
int newtonroot(std::function<arma::vec(arma::vec)> f, arma::vec& x, double const& dx = 1e-6, double const& tol = 1e-12, unsigned int const& max_step = 50);

#endif
