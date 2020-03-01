#ifndef __SINGLE_IMPURITY_ANDERSON_MODEL_H__
#define __SINGLE_IMPURITY_ANDERSON_MODEL_H__

#include <armadillo>
#include <functional>

struct SIAM
{
	using d2d = std::function<double(double)>;

	SIAM(
			d2d							Ed_,
			d2d							E_nuc_,
			arma::vec		const&		bath_,
			arma::vec 		const& 		cpl_,
			double			const&		U_,
			arma::uword 	const& 		n_occ_
	);

	void				set_and_calc(double const& x_);
	double				x;

	void				solve_mf();
	double				n_mf;
	double				E_mf;
	arma::mat			vec_mf;
	arma::vec 			val_mf;

	void				rotate_orb();
	void				subrotate(arma::mat const& vec_sub, double& val_d, arma::vec& vec_d, arma::mat& vec_other, arma::sp_mat& H_other, arma::mat& H_d_other);
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

	void				solve_cisnd();
	arma::mat			H_cisnd();
	arma::mat			n_cisnd();
	arma::vec			val_cisnd;
	arma::mat 			vec_cisnd;

	void				calc_bath();

	void 				calc_Gamma();

	void 				calc_dc();

	d2d					Ed;
	d2d 				E_nuc;
	arma::vec			bath;
	arma::vec 			cpl;
	arma::sp_mat		h; // one-body part of the Hamiltonian
	double				U;
	arma::uword			n_bath;
	arma::uword 		n_occ;
	arma::uword 		n_vir;
	arma::span			span_occ;
	arma::span 			span_vir;
	arma::sp_mat		F(double const& n);
	arma::sp_mat 		F();
	double				n2n(double const& n);


	// data for the last position, used to calculate force and dc
	void				move_new_to_old();
	double				_x;
	arma::vec			_val_cis_sub;
	arma::mat 			_vec_cis_sub;
	arma::vec			_vec_do;
	arma::vec			_vec_dv;
	arma::mat			_vec_o;
	arma::mat			_vec_v;

	// matrix elements
	arma::mat			Pdodv();
	arma::mat			Pdv();
	arma::mat			Pdo();
	arma::mat			Pdob();
	arma::mat			Pjdv();
	arma::mat			Pdvb();
	arma::mat			Pjdo();
	arma::mat			Pab();
	arma::mat			Pji();
	
	arma::mat			Fdo();
	arma::mat			Fdv();
	arma::mat			Fo();
	arma::mat			Fv();
	arma::mat			Fdvb();
	arma::mat			Fjdo();
	arma::mat			Io();
	arma::mat			Iv();

	arma::mat			H_gnd_gnd();
	arma::mat			H_gnd_dodv();
	arma::mat			H_gnd_dob();
	arma::mat			H_gnd_jdv();
	arma::mat			H_gnd_ovov();
	arma::mat			H_gnd_ovob();
	arma::mat			H_gnd_ovjv();

	arma::mat			H_dodv_dodv();
	arma::mat			H_dodv_dob();
	arma::mat			H_dodv_jdv();
	arma::mat			H_dodv_ovov();
	arma::mat			H_dodv_ovob();
	arma::mat			H_dodv_ovjv();

	arma::mat			H_doa_dob();
	arma::mat			H_doa_jdv();
	arma::mat			H_doa_ovov();
	arma::mat			H_doa_ovob();
	arma::mat			H_doa_ovjv();

	arma::mat			H_idv_jdv();
	arma::mat			H_idv_ovov();
	arma::mat			H_idv_ovob();
	arma::mat			H_idv_ovjv();

	arma::mat			H_ovov_ovov();
	arma::mat			H_ovov_ovob();
	arma::mat			H_ovov_ovjv();

	arma::mat			H_ovoa_ovob();
	arma::mat			H_ovoa_ovjv();

	arma::mat			H_oviv_ovjv();

	arma::mat			n_gnd_gnd();
	arma::mat			n_gnd_dodv();
	arma::mat			n_gnd_dob();
	arma::mat			n_gnd_jdv();
	arma::mat			n_gnd_ovov();
	arma::mat			n_gnd_ovob();
	arma::mat			n_gnd_ovjv();

	arma::mat			n_dodv_dodv();
	arma::mat			n_dodv_dob();
	arma::mat			n_dodv_jdv();
	arma::mat			n_dodv_ovov();
	arma::mat			n_dodv_ovob();
	arma::mat			n_dodv_ovjv();

	arma::mat			n_doa_dob();
	arma::mat			n_doa_jdv();
	arma::mat			n_doa_ovov();
	arma::mat			n_doa_ovob();
	arma::mat			n_doa_ovjv();

	arma::mat			n_idv_jdv();
	arma::mat			n_idv_ovov();
	arma::mat			n_idv_ovob();
	arma::mat			n_idv_ovjv();

	arma::mat			n_ovov_ovov();
	arma::mat			n_ovov_ovob();
	arma::mat			n_ovov_ovjv();

	arma::mat			n_ovoa_ovob();
	arma::mat			n_ovoa_ovjv();

	arma::mat			n_oviv_ovjv();
};


#endif
