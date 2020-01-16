#include <TwoPara.h>
#include <gauss.h>
#include <join.h>

using namespace arma;

TwoPara::TwoPara(	PES				E_mpt_,
					PES				E_fil_,
					vec const&		bath_,
					Cpl				cpl_,
					uword const&	n_occ_		):
	E_mpt(E_mpt_), E_fil(E_fil_), bath(bath_), cpl(cpl_), n_occ(n_occ_)
{
	n_bath = bath.n_elem;
	idx_occ = span(0, n_occ-1);
	idx_vir = span(n_occ, n_bath);
	n_vir = n_bath + 1 - n_occ;
	sz_sub = n_occ + n_vir; // size of the CIS subspace plus the ground state
	dE_bath_avg = ( bath.max() - bath.min() ) / (bath.n_elem - 1);

	x = 0;
	H = diagmat( join_cols(vec{0}, bath) );
	Ho = sp_mat(n_occ-1, n_occ-1);
	Hv = sp_mat(n_vir-1, n_vir-1);
}

void TwoPara::set_and_calc(double const& x_) {
	set_and_calc_cis_sub(x_);
	calc_val_cis_bath();
	calc_Gamma();
}

void TwoPara::set_and_calc_cis_sub(double const& x_) {
	x = x_;
	solve_orb();
	rotate_orb();
	solve_cis_sub();
}

void TwoPara::solve_orb() {
	H(0,0) = E_fil(x) - E_mpt(x);
	H(span(1,n_bath), 0) = cpl(x); 
	H(0, span(1,n_bath)) = H(span(1,n_bath), 0).t();

	eig_sym( val_H, vec_H, H );
	ev_n = accu( square( vec_H(0,idx_occ) ) );
	ev_H = accu( val_H(idx_occ) );
}

void TwoPara::rotate_orb() {
	subrotate(vec_H.cols(idx_occ), H, val_do, vec_do, vec_occ, Ho, H_do_occ);
	subrotate(vec_H.cols(idx_vir), H, val_dv, vec_dv, vec_vir, Hv, H_dv_vir);
}

void TwoPara::subrotate(mat const& vec_sub, mat const& H_, double& val_d, vec& vec_d, mat& vec_other, sp_mat& H_bath, mat& H_d_bath) {
	uword sz = vec_sub.n_cols;
	mat q, r;
	vec val_bath;

	// first rotation: separate the Schmidt orbital from the subspace
	mat Q = eye(sz, sz);
	Q.col(0) = vec_sub.row(0).t();
	qr_econ(q,r,Q);

	vec_d = vec_sub * q.col(0);
	vec_other = vec_sub * q.tail_cols(sz-1);
	val_d = as_scalar( vec_d.t() * H_ * vec_d );

	// second rotation: make H diagonal in the bath subspace
	eig_sym( val_bath, q, vec_other.t() * H_ * vec_other);
	vec_other *= q;
	H_bath.zeros();
	H_bath.diag() = val_bath;
	H_d_bath = vec_d.t() * H_ * vec_other;
}

void TwoPara::solve_cis_sub() {
	sp_mat H_doa_dob = speye(n_vir-1, n_vir-1) * (ev_H - val_do) + Hv;
	sp_mat H_doa_jdv = zeros<sp_mat>(n_vir-1, n_occ-1);
	sp_mat H_doa_dov = conv_to<sp_mat>::from( H_dv_vir.t() );
	sp_mat H_idv_jdv = speye(n_occ-1, n_occ-1) * (ev_H + val_dv) - Ho;
	sp_mat H_idv_dov = conv_to<sp_mat>::from( -H_do_occ.t() );
	sp_mat H_dov_dov = conv_to<sp_mat>::from( vec{ev_H - val_do + val_dv} );

	sp_mat H_cis_sub = join<sp_mat>( {
			{ H_doa_dob,     H_doa_jdv,     H_doa_dov },
			{ H_doa_jdv.t(), H_idv_jdv,     H_idv_dov },
			{ H_doa_dov.t(), H_idv_dov.t(), H_dov_dov } } );

	eig_sym( val_cis_sub, vec_cis_sub, conv_to<mat>::from(H_cis_sub) );
}

void TwoPara::calc_val_cis_bath() {
	val_cis_bath = vectorise( ev_H - repmat(Ho.diag(), 1, n_vir-1) + 
			repmat( Hv.diag().t(), n_occ-1, 1) );
}

void TwoPara::calc_Gamma() {
	mat V_adi = vec_cis_sub.t() *
		join<sp_mat>( { { -kron( speye(n_vir-1, n_vir-1), sp_mat(H_do_occ) ) },
						{ kron( sp_mat(H_dv_vir), speye(n_occ-1, n_occ-1) ) },
						{ sp_mat( 1, (n_occ-1)*(n_vir-1) ) } } );
	mat delta = gauss( val_cis_sub, val_cis_bath.as_row(), 5.0*dE_bath_avg );
	Gamma =  2.0 * datum::pi * sum( square(V_adi) % delta, 1 );
}

arma::mat TwoPara::H_(double const& x_) {
	arma::mat h = diagmat( join_cols( vec{E_fil(x_) - E_mpt(x_)}, bath ) );
	h(span(1,n_bath), 0) = cpl(x_); 
	h(0, span(1,n_bath)) = h(span(1,n_bath), 0).t();
	return h;
}

double TwoPara::force_(unsigned int const& state_) {
	if (state_ >= n_occ + n_vir)
		std::cout << "error: state index exceeds boundary" << std::endl;
	double dx = 1e-6;
	if (state_ == 0)
		return - ( accu( eig_sym(H_(x+dx))(idx_occ) ) - ev_H ) / dx;

	TwoPara model_(E_mpt, E_fil, bath, cpl, n_occ);
	model_.set_and_calc_cis_sub(x+dx);
	return - ( model_.val_cis_sub(state_-1) - val_cis_sub(state_-1) ) / dx;
}

vec TwoPara::force_() {
	double dx = 1e-4;
	vec f = zeros(sz_sub);
	f(0) = force_(0);

	TwoPara model_(E_mpt, E_fil, bath, cpl, n_occ);
	model_.set_and_calc_cis_sub(x+dx);
	f.tail(sz_sub-1) = - ( model_.val_cis_sub - val_cis_sub ) / dx;
	
	return f;
}

double TwoPara::force_(double const& x_, unsigned int const& state_) {
	TwoPara model_(E_mpt, E_fil, bath, cpl, n_occ);
	model_.set_and_calc_cis_sub(x_);
	return model_.force_(state_);
}

mat TwoPara::vec_all_rot() {
	return join_r<mat>({vec_do, vec_occ, vec_dv, vec_vir});
}

// the next four functions return occupied orbital indices for subspace 
// basis functions (with respect to the order specified by vec_all_rot())
uvec TwoPara::idx_gnd() {
	return regspace<uvec>(0, n_occ-1);
}

uvec TwoPara::idx_doa(uword const& a) {
	return join_cols( uvec{n_occ+1+a}, regspace<uvec>(1, n_occ-1) );
}

uvec TwoPara::idx_idv(uword const& i) {
	uvec idx = regspace<uvec>(0, n_occ-1);
	idx(i+1) = n_occ;
	return idx;
}

uvec TwoPara::idx_dov() {
	return join_cols( uvec{n_occ}, regspace<uvec>(1, n_occ-1) );
}

// for a row/column index in subspace, return the occupied indices
uvec TwoPara::idx(uword const& n) {
	if (n == 0)
		return idx_gnd();
	if (n == sz_sub-1)
		return idx_dov();
	if (n < n_vir)
		return idx_doa(n-1);
	return idx_idv(n-n_vir);
}

cx_mat TwoPara::dc_() {
	double dx = 1e-4;
	TwoPara model_(E_mpt, E_fil, bath, cpl, n_occ);
	model_.set_and_calc_cis_sub(x+dx);
	mat orb_overlap = vec_all_rot().t() * model_.vec_all_rot();

	mat overlap = zeros(sz_sub, sz_sub);

	for (uword row = 0; row != sz_sub; ++row)
		for (uword col = 0; col != sz_sub; ++col)
			overlap(row, col) = det( orb_overlap(idx(row), idx(col)) );

	return logmat( vec_cis_sub.t() * overlap * model_.vec_cis_sub ) / dx;
}
