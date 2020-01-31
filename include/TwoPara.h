#ifndef __TWO_PARABOLA_H__
#define __TWO_PARABOLA_H__

#include <functional>
#include <armadillo>

struct TwoPara
{
	using d2d = std::function<double(double)>;

	TwoPara(	d2d						E_mpt_,
				d2d						E_fil_,
				arma::vec const&		bath_,
				arma::vec const&		cpl_,
				arma::uword const&		n_occ_		);

	d2d					E_mpt;
	d2d					E_fil;
	d2d					E_imp;
	d2d					dE_mpt;
	d2d					dE_imp;

	arma::vec			bath;
	arma::vec			cpl;

	arma::uword			n_occ;
	arma::uword 		n_vir;
	arma::uword			n_bath;
	arma::uword			sz_rel;
	double				dE_bath_avg;

	arma::span			span_occ;
	arma::span 			span_vir;

	void				set_and_calc(double const& x);
	void				set_and_calc_cis_sub(double const& x);
	double				x;

	void				solve_orb();
	arma::mat			H;
	arma::vec			val_H;
	arma::mat 			vec_H;
	double				ev_n;
	double 				ev_H;

	void				rotate_orb();
	void				subrotate(arma::mat const& vec_sub, double& val_d, arma::vec& vec_d, arma::mat& vec_other, arma::sp_mat& H_bath, arma::mat& H_d_bath);
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

	void				solve_cis_sub();
	arma::vec			val_cis_sub;
	arma::mat 			vec_cis_sub;

	void				calc_val_cis_bath();
	void				calc_Gamma();
	arma::vec			val_cis_bath;
	arma::vec 			Gamma;

	double				E_rel(arma::uword const& state_);
	arma::vec			E_rel();
	double				force(arma::uword const& state_);
	arma::vec			force();
	arma::mat			dc(arma::uword const& sz_);
};


#endif
