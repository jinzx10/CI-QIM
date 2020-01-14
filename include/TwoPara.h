#ifndef __TWO_PARABOLA_H__
#define __TWO_PARABOLA_H__

#include <functional>
#include <armadillo>

struct TwoPara
{
	using PES = std::function<double(double)>;
	using Cpl = std::function<arma::vec(double)>;

	TwoPara(	PES						E_mpt_,
				PES						E_fil_,
				arma::vec const&		bath_,
				Cpl						cpl_,
				arma::uword const&		n_occ_		);

	PES					E_mpt;
	PES					E_fil;
	arma::vec			bath;
	Cpl					cpl;
	arma::uword			n_occ;
	arma::uword 		n_vir;
	arma::uword			n_bath;
	arma::span			idx_occ;
	arma::span 			idx_vir;
	arma::sp_mat		Io;
	arma::sp_mat 		Iv;
	double				dE_bath_avg;

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
	double				val_do;
	double 				val_dv;
	arma::sp_mat		Ho;
	arma::sp_mat 		Hv;
	arma::mat			H_do_occ;
	arma::mat 			H_dv_vir;

	void				solve_cis_sub();
	arma::vec			val_cis_sub;
	arma::mat 			vec_cis_sub;

	void				calc_val_cis_bath();
	void				calc_Gamma();
	arma::vec			val_cis_bath;
	arma::vec 			Gamma;

	// calculate without changing contents in the class
	arma::mat			H_(double const& x_);
	double				force_(unsigned int const& state_);
	double				force_(double const& x_, unsigned int const& state_);
};

void subrotate(arma::subview<double> const& vec_sub, arma::mat const& H, double& val_d, arma::sp_mat& H_bath, arma::mat& H_d_bath);

#endif
