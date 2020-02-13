#ifndef __SINGLE_IMPURITY_ANDERSON_MODEL_H__
#define __SINGLE_IMPURITY_ANDERSON_MODEL_H__

#include <armadillo>

struct SIAM
{
	SIAM(
			arma::vec const& bath_,
			arma::vec const& cpl_,
			double const& U_,
			arma::uword const& n_elec_ // should be even
	);

	arma::vec bath;
	arma::vec cpl;
	double U;
	arma::uword n_elec;

	arma::mat eigvec_mf;
	arma::mat eigval_mf;

	void solve_mf();

};


#endif
