#ifndef __SINGLE_IMPURITY_ANDERSON_MODEL_H__
#define __SINGLE_IMPURITY_ANDERSON_MODEL_H__

#include <armadillo>
#include <functional>

struct SIAM
{
	using d2d = std::function<double(double)>;

	SIAM(
			d2d                         E_imp_,
			d2d                         E_nuc_,
			arma::vec       const&      bath_,
			arma::vec       const&      cpl_,
			double          const&      U_,
			arma::uword     const&      n_occ_,
			arma::uword     const&      sz_sub_,
			double          const&      x_
	);

	void                set_and_calc(double const& x_);
	double              x;

	// mean-field
	void                solve_mf();
	double              n_mf;
	double              E_mf;
	arma::mat           vec_mf;
	arma::vec           val_mf;

	// orbital rotation
	void                rotate_orb();
	double              F_dodo;
	double              F_dvdv;
	arma::vec           vec_do;
	arma::vec           vec_dv;
	arma::mat           vec_o; // occupied bath orbitals
	arma::mat           vec_v; // virtual bath orbitals
	arma::sp_mat        F_ij; // diagonal 
	arma::sp_mat        F_ab; // diagonal

	// adjust sign
	void                adj_orb_sign();

	// basic elements for matrix
	void                calc_basic_elem();
	double              P_dodo;
	double              P_dvdv;
	double              P_dodv;
	arma::mat           F_dvb;
	arma::mat           F_doj;

	// selective CIS-ND
	void                solve_cisnd();
	arma::sp_mat        H_cisnd();
	arma::sp_mat        N_cisnd();
	arma::vec           E_cisnd;
	arma::vec           n_cisnd;
	arma::vec           val_cisnd;
	arma::mat           vec_cisnd;

	void                calc_force();
	arma::vec           F_cisnd;
	double              F_nucl;

	void                calc_Gamma_rlx();
	arma::vec           E_bath();
	arma::sp_mat        V_cpl();
	arma::vec           Gamma_rlx;

	// derivative coupling of a subspace of adiabats
	void                calc_dc_adi();
	arma::mat           dc_adi;
	arma::mat           ovl_sub_raw;

	// data for the last position, used to calculate force and dc
	void                move_new_to_old();
	double              _x;
	arma::vec           _vec_do;
	arma::vec           _vec_dv;
	arma::mat           _vec_o;
	arma::mat           _vec_v;
	arma::vec           _val_cisnd;
	arma::mat           _vec_cisnd;

	d2d                 E_imp;
	d2d                 E_nuc;
	arma::vec           bath;
	arma::vec           cpl;
	arma::sp_mat        h; // one-body part of the Hamiltonian
	double              U;
	arma::uword         n_bath;
	arma::uword         n_occ;
	arma::uword         n_vir;
	arma::span          span_occ;
	arma::span          span_vir;
	arma::uword         sz_sub; // size of subspace adiabats for dynamics
	arma::span          span_sub;
	arma::uword         sz_cisnd;
	arma::uword         sz_jb;
	double              dE_avg;
	arma::mat           F();
	arma::mat           F(double const& n);
	//double              n2n(double const& n);
	arma::sp_mat        Io();
	arma::sp_mat        Iv();

	// matrix elements
	// H_cisnd
	arma::sp_mat        H_gnd_gnd();
	arma::sp_mat        H_gnd_dodv();
	arma::sp_mat        H_gnd_dob();
	arma::sp_mat        H_gnd_jdv();
	arma::sp_mat        H_gnd_ovov();
	arma::sp_mat        H_gnd_ovob();
	arma::sp_mat        H_gnd_ovjv();

	arma::sp_mat        H_dodv_dodv();
	arma::sp_mat        H_dodv_dob();
	arma::sp_mat        H_dodv_jdv();
	arma::sp_mat        H_dodv_ovov();
	arma::sp_mat        H_dodv_ovob();
	arma::sp_mat        H_dodv_ovjv();

	arma::sp_mat        H_doa_dob();
	arma::sp_mat        H_doa_jdv();
	arma::sp_mat        H_doa_ovov();
	arma::sp_mat        H_doa_ovob();
	arma::sp_mat        H_doa_ovjv();

	arma::sp_mat        H_idv_jdv();
	arma::sp_mat        H_idv_ovov();
	arma::sp_mat        H_idv_ovob();
	arma::sp_mat        H_idv_ovjv();

	arma::sp_mat        H_ovov_ovov();
	arma::sp_mat        H_ovov_ovob();
	arma::sp_mat        H_ovov_ovjv();

	arma::sp_mat        H_ovoa_ovob();
	arma::sp_mat        H_ovoa_ovjv();

	arma::sp_mat        H_oviv_ovjv();


	// V_cpl
	arma::sp_mat        H_doa_jb();
	arma::sp_mat        H_idv_jb();

	arma::sp_mat        H_ovoa_ovjb();
	arma::sp_mat        H_oviv_ovjb();


	// N_cisnd
	arma::sp_mat        N_gnd_gnd();
	arma::sp_mat        N_gnd_dodv();
	arma::sp_mat        N_gnd_dob();
	arma::sp_mat        N_gnd_jdv();
	arma::sp_mat        N_gnd_ovov();
	arma::sp_mat        N_gnd_ovob();
	arma::sp_mat        N_gnd_ovjv();

	arma::sp_mat        N_dodv_dodv();
	arma::sp_mat        N_dodv_dob();
	arma::sp_mat        N_dodv_jdv();
	arma::sp_mat        N_dodv_ovov();
	arma::sp_mat        N_dodv_ovob();
	arma::sp_mat        N_dodv_ovjv();

	arma::sp_mat        N_doa_dob();
	arma::sp_mat        N_doa_jdv();
	arma::sp_mat        N_doa_ovov();
	arma::sp_mat        N_doa_ovob();
	arma::sp_mat        N_doa_ovjv();

	arma::sp_mat        N_idv_jdv();
	arma::sp_mat        N_idv_ovov();
	arma::sp_mat        N_idv_ovob();
	arma::sp_mat        N_idv_ovjv();

	arma::sp_mat        N_ovov_ovov();
	arma::sp_mat        N_ovov_ovob();
	arma::sp_mat        N_ovov_ovjv();

	arma::sp_mat        N_ovoa_ovob();
	arma::sp_mat        N_ovoa_ovjv();

	arma::sp_mat        N_oviv_ovjv();
};

void subrotate(arma::mat const& vec_sub, arma::vec& vec_d, arma::mat& vec_other, arma::mat const& H, double& H_d, arma::sp_mat& H_other);

void zeyu_sign(arma::mat const& _vecs, arma::mat& vecs, arma::mat const& S = arma::mat{});

arma::mat S_exact(arma::vec const& _vec_do, arma::mat const& _vec_o, arma::vec const& _vec_dv, arma::mat const& _vec_v, arma::vec const& vec_do, arma::mat const& vec_o, arma::vec const& vec_dv, arma::mat const& vec_v);

#endif
