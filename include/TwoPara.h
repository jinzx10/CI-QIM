#ifndef __TWO_PARABOLA_H__
#define __TWO_PARABOLA_H__

#include <functional>
#include <armadillo>
#include <string>

struct TwoPara
{
	using d2d = std::function<double(double)>;

	TwoPara(	
			d2d							E_mpt_,
			d2d							E_fil_,
			arma::vec		const&		bath_,
			arma::vec		const&		cpl_,
			arma::uword		const&		n_occ_,
			arma::uword 	const&		sz_sub_,
			double			const&		x_init_
	);

	d2d					E_nuc;
	d2d					E_imp;
	d2d					dE_nuc;
	d2d					dE_imp;

	arma::vec			bath;
	arma::vec			cpl;
	arma::uword			n_bath;
	double				dE_bath_avg;

	arma::uword			n_occ;
	arma::uword 		n_vir;
	arma::span			span_occ;
	arma::span 			span_vir;
	arma::uword			sz_cis;

	arma::uword			sz_sub;

	void				set_and_calc(double const& x);
	double				x;

	void				solve_orb();
	arma::mat			H;
	arma::vec			val_H;
	arma::mat 			vec_H;
	double				ev_n;
	double 				ev_H;

	void				rotate_orb();
	arma::vec			vec_do;
	arma::vec			vec_dv;
	arma::mat			vec_o; // occupied bath orbitals
	arma::mat			vec_v; // virtual bath orbitals
	double				H_dodo;
	double 				H_dvdv;
	arma::sp_mat		H_ij; // diagonal 
	arma::sp_mat 		H_ab; // diagonal

	// adjust sign
	void				adj_orb_sign();

	// matrix elements of H in the rotated orbital basis
	void				calc_basic_elem();
	arma::mat			H_doj;
	arma::mat 			H_dvb;

	// selective CIS
	void				solve_slt_cis();
	arma::vec			val_slt_cis;
	arma::mat 			vec_slt_cis;

	void				calc_force();
	double				F_gnd;
	arma::vec			F_slt_cis;

	void				calc_Gamma_rlx();
	arma::vec			E_bath();
	arma::vec 			Gamma_rlx;

	void				calc_dc_adi();
	arma::mat			dc_adi;
	arma::mat 			ovl_sub_raw; // raw overlap matrix of subspace adiabats (before Lowdin)
	//arma::mat           S; // basis overlap matrix

	arma::vec			E_sub();
	arma::vec			F_sub();

	// data for the last position, used to calculate force and dc
	void				move_new_to_old();
	double				_x;
	arma::vec			_val_slt_cis;
	arma::mat 			_vec_slt_cis;
	arma::vec			_vec_do;
	arma::vec			_vec_dv;
	arma::mat			_vec_o;
	arma::mat			_vec_v;
};


void subrotate(arma::mat const& vec_sub, arma::vec& vec_d, arma::mat& vec_other, arma::mat const& H, double& H_d, arma::sp_mat& H_other);

void zeyu_sign(arma::mat const& _vecs, arma::mat& vecs, arma::mat const& S = arma::mat{});

arma::mat S_exact(arma::vec const& _vec_do, arma::mat const& _vec_o, arma::vec const& _vec_dv, arma::mat const& _vec_v, arma::vec const& vec_do, arma::mat const& vec_o, arma::vec const& vec_dv, arma::mat const& vec_v);


#endif
