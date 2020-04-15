#ifndef __SINGLE_IMPURITY_ANDERSON_MODEL_H__
#define __SINGLE_IMPURITY_ANDERSON_MODEL_H__

#include <armadillo>
#include <functional>

struct SIAM
{
	using d2d = std::function<double(double)>;

	SIAM(
			d2d							E_imp_,
			d2d							E_nuc_,
			arma::vec		const&		bath_,
			arma::vec 		const& 		cpl_,
			double			const&		U_,
			arma::uword 	const& 		n_occ_,
			arma::uword		const&		sz_sub_
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
	arma::sp_mat		F_o; // diagonal 
	arma::sp_mat 		F_v; // diagonal
	arma::mat			F_do_o;
	arma::mat 			F_dv_v;

	// selective CIS-ND
	void				solve_cisnd();
	arma::mat			H_cisnd();
	arma::mat			N_cisnd();
	arma::vec			E_cisnd;
	arma::vec			n_cisnd;
	arma::vec			val_cisnd;
	arma::mat 			vec_cisnd;
	arma::mat			coef;

	void				calc_bath();
	void 				calc_Gamma();

	// derivative coupling of a subspace of adiabats
	arma::uword			sz_sub; // size of subspace adiabats for running dynamics
	void 				calc_dc_adi();
	arma::mat			dc_adi;

	// data for the last position, used to calculate force and dc
	void				move_new_to_old();
	double				_x;
	arma::vec			_vec_do;
	arma::vec			_vec_dv;
	arma::mat			_vec_o;
	arma::mat			_vec_v;
	arma::vec			_val_cisnd;
	arma::mat 			_coef; // part of vec_cisnd

	d2d					E_imp;
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
	arma::span			span_sub;
	arma::sp_mat		F(double const& n);
	arma::sp_mat 		F();
	double				n2n(double const& n);

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

	arma::mat			N_gnd_gnd();
	arma::mat			N_gnd_dodv();
	arma::mat			N_gnd_dob();
	arma::mat			N_gnd_jdv();
	arma::mat			N_gnd_ovov();
	arma::mat			N_gnd_ovob();
	arma::mat			N_gnd_ovjv();

	arma::mat			N_dodv_dodv();
	arma::mat			N_dodv_dob();
	arma::mat			N_dodv_jdv();
	arma::mat			N_dodv_ovov();
	arma::mat			N_dodv_ovob();
	arma::mat			N_dodv_ovjv();

	arma::mat			N_doa_dob();
	arma::mat			N_doa_jdv();
	arma::mat			N_doa_ovov();
	arma::mat			N_doa_ovob();
	arma::mat			N_doa_ovjv();

	arma::mat			N_idv_jdv();
	arma::mat			N_idv_ovov();
	arma::mat			N_idv_ovob();
	arma::mat			N_idv_ovjv();

	arma::mat			N_ovov_ovov();
	arma::mat			N_ovov_ovob();
	arma::mat			N_ovov_ovjv();

	arma::mat			N_ovoa_ovob();
	arma::mat			N_ovoa_ovjv();

	arma::mat			N_oviv_ovjv();
};

void zeyu_sign(arma::mat const& vecs_old, arma::mat& vecs_new, arma::mat const& S = arma::mat{});

arma::mat calc_dc(arma::mat const& _coef, arma::mat const& coef, double const& dx, arma::mat const& S = arma::mat{});

arma::mat S_exact(arma::vec const& vec_do_, arma::mat const& vec_occ_, arma::vec const& vec_dv_, arma::mat const& vec_vir_, arma::vec const& vec_do, arma::mat const& vec_occ, arma::vec const& vec_dv, arma::mat const& vec_vir);

#endif
