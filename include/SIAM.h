#ifndef __SINGLE_IMPURITY_ANDERSON_MODEL_H__
#define __SINGLE_IMPURITY_ANDERSON_MODEL_H__

#include <armadillo>

struct SIAM
{
	SIAM(
			arma::vec const& bath_,
			arma::vec const& cpl_,
			double const& U_,
			arma::uword const& n_occ_
	);

	void set_and_calc(double const& Ed);
	void solve_mf();

	double n_mf;
	double E_mf;


	private:

	double Ed;
	arma::vec bath;
	arma::vec cpl;
	arma::mat h; // one-body part of the Hamiltonian
	double U;
	arma::uword n_bath;
	arma::uword n_occ;
	arma::uword n_vir;
	arma::mat F(double const& n);
	double n2n(double const& n);

	arma::mat vec_mf;
	arma::vec val_mf;

};


#endif
