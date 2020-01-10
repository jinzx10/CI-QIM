#ifndef __TWO_PARABOLA_H__
#define __TWO_PARABOLA_H__

#include <functional>
#include <armadillo>

struct TwoPara
{
	using PES = std::function<double(double)>;
	using Cpl = std::function<arma::vec(double)>;

	TwoPara( PES E_mpt_,
			PES E_fil_,
			arma::vec const& bath_,
			Cpl cpl_,
			arma::uword const& n_elec_);

	PES E_mpt;
	PES E_fil;
	arma::vec bath;
	Cpl cpl;
	arma::uword n_elec;

	arma::uword	n_bath;
	arma::span idx_occ;
	arma::span idx_vir;
	arma::uword n_occ;
	arma::uword n_vir;

	void solve_orb(double const& x);
	double x;
	arma::mat H;
	arma::vec val_H;
	arma::mat vec_H;
	double ev_n;
	double ev_H;

	void first_rotate(); // separate do/dv from the occ/vir subspaces
	void second_rotate(); // rotate bath states such that H is diagonal in bath subspaces
	arma::vec vec_do;
	arma::vec vec_dv;
	double val_do;
	double val_dv;
	arma::mat vec_bath_occ;
	arma::mat vec_bath_vir;
	arma::vec val_bath_occ;
	arma::vec val_bath_vir;

	void solve_cis_sub();
	arma::vec val_cis_sub;
	arma::mat vec_cis_sub;

	void calc_cis_bath();
	arma::vec E_cis_bath;
	arma::vec Gamma;


	arma::sp_mat V_sub_bath();
	arma::sp_mat H_doa_jb;
	arma::sp_mat H_idv_jb;
	arma::sp_mat H_dov_jb;

};

#endif
