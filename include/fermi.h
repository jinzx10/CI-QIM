#ifndef __FERMI_FUNCTION_H__
#define __FERMI_FUNCTION_H__

#include <armadillo>

double fermi(double const& E, double const& mu, double const& kT);
arma::vec fermi(arma::vec const& E, double const& mu, double const& kT);

#endif
