#ifndef __SINGLE_IMPURITY_ANDERSON_MODEL_H__
#define __SINGLE_IMPURITY_ANDERSON_MODEL_H__

#include <armadillo>
#include <functional>

struct SIAM
{
	using d2d = std::function<double(double)>;

	SIAM(
			d2d		Ed_,
			d2d		E_nuc_,
			arma::vec const& bath_,
			arma::vec const& cpl_,
			double const& U_,
			arma::uword const& n_occ_
	);

	void set_and_calc(double const& x_);
	double x;

	void solve_mf();
	double n_mf;
	double E_mf;
	arma::mat vec_mf;
	arma::vec val_mf;

	void rotate_orb();
	void subrotate(arma::mat const& vec_sub, double& val_d, arma::vec& vec_d, arma::mat& vec_other, arma::sp_mat& H_other, arma::mat& H_d_other);
	// o(v) stands for occupied (virtual) except do(dv)
	double				val_do;
	double 				val_dv;
	arma::vec			vec_do;
	arma::vec			vec_dv;
	arma::mat			vec_o;
	arma::mat			vec_v;
	arma::sp_mat		H_o; // diagonal 
	arma::sp_mat 		H_v; // diagonal
	arma::mat			H_do_o;
	arma::mat 			H_dv_v;

	void solve_cisnd();

	void calc_bath();
	void calc_Gamma();
	void calc_dc();

	d2d Ed;
	d2d E_nuc;
	arma::vec bath;
	arma::vec cpl;
	arma::sp_mat h; // one-body part of the Hamiltonian
	double U;
	arma::uword n_bath;
	arma::uword n_occ;
	arma::uword n_vir;
	arma::span span_occ;
	arma::span span_vir;
	arma::sp_mat F(double const& n);
	arma::sp_mat F();
	double n2n(double const& n);


	// data for the last position, used to calculate force and dc
	void				move_new_to_old();
	double				_x;
	arma::vec			_val_cis_sub;
	arma::mat 			_vec_cis_sub;
	arma::vec			_vec_do;
	arma::vec			_vec_dv;
	arma::mat			_vec_o;
	arma::mat			_vec_v;
};


#endif
