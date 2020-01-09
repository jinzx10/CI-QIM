#ifndef __FIND_CHEMICAL_POTENTIAL_H__
#define __FIND_CHEMICAL_POTENTIAL_H__

#include <armadillo>

double findmu(arma::vec const& E, arma::uword const& n, double const& kT);

#endif
