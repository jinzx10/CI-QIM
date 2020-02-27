#ifndef __TWO_PARABOLA_H__
#define __TWO_PARABOLA_H__

#include <functional>
#include <armadillo>
#include <string>

struct TwoPara
{
	using d2d = std::function<double(double)>;

	TwoPara(	
			d2d						E_mpt_,
			d2d						E_fil_,
			arma::vec const&		bath_,
			arma::vec const&		cpl_,
			arma::uword const&		n_occ_,
			double const&			x_init_
	);

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
	//void				set_and_calc_cis_sub(double const& x);
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
	arma::vec			val_cis_bath;

	void				calc_Gamma(arma::uword const& sz_);
	arma::vec 			Gamma;

	void				calc_force();
	arma::vec			force;

	void				calc_dc(arma::uword const& sz_, std::string const& = "approx");
	arma::mat			dc;

	double				E_rel(arma::uword const& state_);
	arma::vec			E_rel();

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


void adj_phase(arma::mat const& vecs_old, arma::mat& vecs_new);

arma::mat ovl(arma::vec const& vec_do_, arma::mat const& vec_occ_, arma::vec const& vec_dv_, arma::mat const& vec_vir_, arma::vec const& vec_do, arma::mat const& vec_occ, arma::vec const& vec_dv, arma::mat const& vec_vir, std::string const& method);

arma::vec det12(arma::mat const& A, arma::uvec const& idx, arma::mat const& vecs, char const& rc = 'c');
arma::vec det3(arma::mat const& A, arma::uword const& i, arma::mat const& u, arma::uvec const& idx, arma::mat const& v, char const& rc = 'c');

#endif
