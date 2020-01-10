#include "../include/TwoPara.h"

TwoPara::TwoPara( PES E_mpt_,
			PES E_fil_,
			arma::vec const& bath_,
			Cpl cpl_,
			arma::uword const& n_elec_):
	E_mpt(E_mpt_), E_fil(E_fil_), bath(bath_), cpl(cpl_), n_elec(n_elec_)
{
	n_bath = bath.n_elem;
	idx_occ = arma::span(0, n_elec-1);
	idx_vir = arma::span(n_elec, n_bath);
	n_occ = n_elec;
	n_vir = n_bath + 1 - n_occ;

	x = 0;
	H = arma::diagmat( arma::join_cols(arma::vec{0}, bath) );
}

void TwoPara::solve_orb(double const& x_) {
	x = x_;

	H(0,0) = E_fil(x) - E_mpt(x);
	H(arma::span(1,n_bath), 0) = cpl(x); 
	H(0, arma::span(1,n_bath)) = H(arma::span(1,n_bath), 0).t();

	arma::eig_sym( val_H, vec_H, H );
	ev_n = arma::accu( arma::square( vec_H(0,idx_occ) ) );
	ev_H = arma::accu( val_H(idx_occ) );
}

void TwoPara::first_rotate() {
	vec_do = vec_H.cols(idx_occ) * vec_H(0, idx_occ).t() / std::sqrt(ev_n);
	vec_dv = vec_H.cols(idx_vir) * vec_H(0, idx_vir).t() / std::sqrt(1.0-ev_n);
	val_do = arma::as_scalar( vec_do.t() * H * vec_do );
	val_dv = arma::as_scalar( vec_dv.t() * H * vec_dv );

	arma::mat q,r;
	arma::mat Q_occ = arma::eye(n_occ, n_occ);
	Q_occ.col(0) = vec_H(0, idx_occ).t();
	arma::qr_econ(q, r, Q_occ);
	vec_bath_occ = vec_H(arma::span::all, idx_occ) * q.tail_cols(n_occ-1);

	arma::mat Q_vir = arma::eye(n_vir, n_vir);
	Q_vir.col(0) = vec_H(0, idx_vir).t();
	arma::qr_econ(q, r, Q_vir);
	vec_bath_vir = vec_H(arma::span::all, idx_vir) * q.tail_cols(n_vir-1);
}

void TwoPara::second_rotate() {
	arma::mat q;
	arma::eig_sym( val_bath_occ, q, vec_bath_occ.t() * H * vec_bath_occ );
	vec_bath_occ *= q;
	arma::eig_sym( val_bath_vir, q, vec_bath_vir.t() * H * vec_bath_vir );
	vec_bath_vir *= q;
}

void TwoPara::solve_cis_sub() {
	arma::sp_mat H_doa_dob = arma::speye(n_vir-1, n_vir-1) * (ev_H - val_do);
	H_doa_dob.diag() += val_bath_vir;
	arma::sp_mat H_doa_jdv = arma::zeros<arma::sp_mat>(n_vir-1, n_occ-1);
	arma::sp_mat H_doa_dov = arma::conv_to<arma::sp_mat>::from( vec_bath_vir.t() * H * vec_dv );
	arma::sp_mat H_idv_jdv = arma::speye(n_occ-1, n_occ-1) * (ev_H + val_dv);
	H_idv_jdv.diag() -= val_bath_occ;
	arma::sp_mat H_idv_dov = arma::conv_to<arma::sp_mat>::from( - ( vec_do.t() * H * vec_bath_occ ).t() );
	arma::sp_mat H_dov_dov = arma::conv_to<arma::sp_mat>::from( arma::vec{ev_H - val_do + val_dv} );

	arma::sp_mat H_cis_sub = arma::join_cols(
			arma::join_rows( H_doa_dob,     H_doa_jdv,     H_doa_dov ),
			arma::join_rows( H_doa_jdv.t(), H_idv_jdv,     H_idv_dov ),
			arma::join_rows( H_doa_dov.t(), H_idv_dov.t(), H_dov_dov ) );

	arma::eig_sym( val_cis_sub, vec_cis_sub, arma::conv_to<arma::mat>::from(H_cis_sub) );
}

void TwoPara::calc_cis_bath() {

}
