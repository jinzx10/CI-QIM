#ifndef __SKELETON_1D_H__
#define __SKELETON_1D_H__

#include "armadillo"

class Skeleton_1d
{
public:
	Skeleton_1d() {}

	virtual double get_energy(arma::vec const& x, arma::uword const& state);
	virtual arma::vec get_energies(arma::vec const& x);
	virtual double get_gradient(arma::vec const& x, arma::uword const& state);
	virtual arma::mat get_drvcpl(arma::vec const& x);
	virtual arma::uword sz_elec();
};

#endif
