#include "../include/TwoPara.h"
#include "../include/gauss.h"

using namespace arma;

TwoPara::TwoPara( PES E_mpt_,
			PES E_fil_,
			vec const& bath_,
			Cpl cpl_,
			uword const& n_elec_):
	E_mpt(E_mpt_), E_fil(E_fil_), bath(bath_), cpl(cpl_), n_elec(n_elec_)
{
	n_bath = bath.n_elem;
	idx_occ = span(0, n_elec-1);
	idx_vir = span(n_elec, n_bath);
	n_occ = n_elec;
	n_vir = n_bath + 1 - n_occ;

	x = 0;
	H = diagmat( join_cols(vec{0}, bath) );

	Io = speye(n_occ-1, n_occ-1);
	Iv = speye(n_vir-1, n_vir-1);
	Ho = sp_mat(n_occ-1, n_occ-1);
	Hv = sp_mat(n_vir-1, n_vir-1);
}

void TwoPara::solve_orb(double const& x_) {
	x = x_;

	H(0,0) = E_fil(x) - E_mpt(x);
	H(span(1,n_bath), 0) = cpl(x); 
	H(0, span(1,n_bath)) = H(span(1,n_bath), 0).t();

	eig_sym( val_H, vec_H, H );
	ev_n = accu( square( vec_H(0,idx_occ) ) );
	ev_H = accu( val_H(idx_occ) );
}

void TwoPara::rotate_orb() {
	// first rotation: separate do/dv from the occ/vir subspaces
	vec_do = vec_H.cols(idx_occ) * vec_H(0, idx_occ).t() / std::sqrt(ev_n);
	vec_dv = vec_H.cols(idx_vir) * vec_H(0, idx_vir).t() / std::sqrt(1.0-ev_n);
	val_do = as_scalar( vec_do.t() * H * vec_do );
	val_dv = as_scalar( vec_dv.t() * H * vec_dv );

	mat q,r;
	mat Q_occ = eye(n_occ, n_occ);
	Q_occ.col(0) = vec_H(0, idx_occ).t();
	qr_econ(q, r, Q_occ);
	mat vec_bath_occ = vec_H(span::all, idx_occ) * q.tail_cols(n_occ-1);

	mat Q_vir = eye(n_vir, n_vir);
	Q_vir.col(0) = vec_H(0, idx_vir).t();
	qr_econ(q, r, Q_vir);
	mat vec_bath_vir = vec_H(span::all, idx_vir) * q.tail_cols(n_vir-1);

	// second rotation: rotate bath states such that H is diagonal in bath subspaces
	vec val;
	eig_sym( val, q, vec_bath_occ.t() * H * vec_bath_occ );
	vec_bath_occ *= q;
	Ho.diag() = val;

	eig_sym( val, q, vec_bath_vir.t() * H * vec_bath_vir );
	vec_bath_vir *= q;
	Hv.diag() = val;

	H_do_occ = vec_do.t() * H * vec_bath_occ;
	H_dv_vir = vec_dv.t() * H * vec_bath_vir;
}



void TwoPara::solve_cis_sub() {
	sp_mat H_doa_dob = Iv * (ev_H - val_do) + Hv;
	sp_mat H_doa_jdv = zeros<sp_mat>(n_vir-1, n_occ-1);
	sp_mat H_doa_dov = conv_to<sp_mat>::from( H_dv_vir.t() );
	sp_mat H_idv_jdv = Io * (ev_H + val_dv) - Ho;
	sp_mat H_idv_dov = conv_to<sp_mat>::from( -H_do_occ.t() );
	sp_mat H_dov_dov = conv_to<sp_mat>::from( vec{ev_H - val_do + val_dv} );

	sp_mat H_cis_sub = join_cols(
			join_rows( H_doa_dob,     H_doa_jdv,     H_doa_dov ),
			join_rows( H_doa_jdv.t(), H_idv_jdv,     H_idv_dov ),
			join_rows( H_doa_dov.t(), H_idv_dov.t(), H_dov_dov ) );

	eig_sym( val_cis_sub, vec_cis_sub, conv_to<mat>::from(H_cis_sub) );
}

void TwoPara::calc_cis_bath() {
	E_cis_bath = kron(Iv, Io) * ev_H - kron(Iv, Ho) + kron(Hv, Io);
	
	sp_mat H_doa_jb = -kron( Iv, sp_mat(H_do_occ) );
	sp_mat H_idv_jb = kron( sp_mat(H_dv_vir), Io );
	sp_mat H_dov_jb = sp_mat( 1, (n_occ-1)*(n_vir-1) );

	mat V_adi = vec_cis_sub.t() * join_cols(H_doa_jb, H_idv_jb, H_dov_jb);
	Gamma =  2 * datum::pi * sum( square(V_adi) % 
			gauss(val_cis_sub, E_cis_bath.t(), 5.0/(bath(1)-bath(0))), 2 );
}



