#include <TwoPara.h>
#include <dc.h>
#include <auxmath.h>
#include <arma_helper.h>

using namespace arma;

TwoPara::TwoPara(	d2d				E_mpt_,
					d2d				E_fil_,
					vec const&		bath_,
					vec const&		cpl_,
					uword const&	n_occ_		):
	E_mpt(E_mpt_), E_fil(E_fil_), bath(bath_), cpl(cpl_), n_occ(n_occ_)
{
	E_imp = [this] (double const& x_) { return E_fil(x_) - E_mpt(x_); };
	dE_mpt = grad(E_mpt);
	dE_imp = grad(E_imp);

	n_bath = bath.n_elem;
	n_vir = n_bath + 1 - n_occ;
	sz_rel = n_occ + n_vir; // size of the relevant subspace (ground state + selected CIS)
	dE_bath_avg = ( bath.max() - bath.min() ) / (bath.n_elem - 1);

	span_occ = span(0, n_occ-1);
	span_vir = span(n_occ, n_bath);

	H = diagmat(join_cols(vec{0}, bath));
	H(span(1, n_bath), 0) = cpl;
	H(0, span(1, n_bath)) = cpl.t();
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
	H(0,0) = E_imp(x);
	eig_sym( val_H, vec_H, H );
	ev_n = accu( square( vec_H(0, span_occ) ) );
	ev_H = accu( val_H(span_occ) );
}

void TwoPara::rotate_orb() {
	subrotate(vec_H.cols(span_occ), val_do, vec_do, vec_o, H_o, H_do_o);
	subrotate(vec_H.cols(span_vir), val_dv, vec_dv, vec_v, H_v, H_dv_v);
}

void TwoPara::subrotate(mat const& vec_sub, double& val_d, vec& vec_d, mat& vec_other, sp_mat& H_other, mat& H_d_other) {
	uword sz = vec_sub.n_cols;

	// first rotation: separate the Schmidt orbital from the subspace
	mat Q = eye(sz, sz);
	Q.col(0) = vec_sub.row(0).t();
	mat q, r;
	qr_econ(q, r, Q);

	vec_d = vec_sub * q.col(0);
	vec_other = vec_sub * q.tail_cols(sz-1);
	val_d = as_scalar( vec_d.t() * H * vec_d );

	// second rotation: make H diagonal in the "other" subspace
	vec val_other;
	eig_sym( val_other, q, vec_other.t() * H * vec_other);
	vec_other *= q;
	H_other.zeros(sz-1, sz-1);
	H_other.diag() = val_other;
	H_d_other= vec_d.t() * H * vec_other;
}

void TwoPara::solve_cis_sub() {
	sp_mat H_dov_dov = conv_to<sp_mat>::from( vec{ev_H - val_do + val_dv} );
	sp_mat H_dov_dob = conv_to<sp_mat>::from( H_dv_v );
	sp_mat H_dov_jdv = conv_to<sp_mat>::from( -H_do_o );
	sp_mat H_doa_dob = speye(n_vir-1, n_vir-1) * (ev_H - val_do) + H_v;
	sp_mat H_doa_jdv = zeros<sp_mat>(n_vir-1, n_occ-1);
	sp_mat H_idv_jdv = speye(n_occ-1, n_occ-1) * (ev_H + val_dv) - H_o;

	sp_mat H_cis_sub = join<sp_mat>( {
			{ H_dov_dov,     H_dov_dob,     H_dov_jdv},
			{ H_dov_dob.t(), H_doa_dob,     H_doa_jdv},
			{ H_dov_jdv.t(), H_doa_jdv.t(), H_idv_jdv} } );

	eig_sym( val_cis_sub, vec_cis_sub, conv_to<mat>::from(H_cis_sub) );
}

void TwoPara::calc_val_cis_bath() {
	val_cis_bath = vectorise( ev_H - repmat(H_o.diag(), 1, n_vir-1) + 
			repmat( H_v.diag().t(), n_occ-1, 1) );
}

void TwoPara::calc_Gamma() {
	mat V_adi = vec_cis_sub.t() * join<sp_mat>( {
			{ sp_mat( 1, (n_occ-1)*(n_vir-1) ) },
			{ -kron( speye(n_vir-1, n_vir-1), sp_mat(H_do_o) ) },
			{ kron( sp_mat(H_dv_v), speye(n_occ-1, n_occ-1) ) }
	});
	mat delta = gauss( val_cis_sub, val_cis_bath.as_row(), 5.0*dE_bath_avg );
	Gamma = 2.0 * datum::pi * sum( square(V_adi) % delta, 1 );
}

void TwoPara::calc_Gamma(uword const& sz) {
	mat V_adi = vec_cis_sub.head_cols(sz).t() * join<sp_mat>( {
			{ sp_mat( 1, (n_occ-1)*(n_vir-1) ) },
			{ -kron( speye(n_vir-1, n_vir-1), sp_mat(H_do_o) ) },
			{ kron( sp_mat(H_dv_v), speye(n_occ-1, n_occ-1) ) }
	});
	mat delta = gauss( val_cis_sub.head(sz), val_cis_bath.as_row(),
			5.0*dE_bath_avg );
	Gamma = 2.0 * datum::pi * sum( square(V_adi) % delta, 1 );
}

double TwoPara::E_rel(uword const& state_) {
	return (state_) ? val_cis_sub(state_-1) : ev_H;
}

vec TwoPara::E_rel() {
	return join_cols( vec{ev_H}, val_cis_sub );
}

double TwoPara::force(uword const& state_) {
	if (state_ == 0)
		return - dE_imp(x) * ev_n - dE_mpt(x);

	double dx = 1e-4;
	TwoPara model_(E_mpt, E_fil, bath, cpl, n_occ);
	model_.set_and_calc_cis_sub(x+dx);
	return - ( model_.val_cis_sub(state_-1) - val_cis_sub(state_-1) ) / dx 
		- dE_mpt(x);
}

vec TwoPara::force() {
	double dx = 1e-4;
	vec f = zeros(sz_rel);
	f(0) = force(0);

	TwoPara model_(E_mpt, E_fil, bath, cpl, n_occ);
	model_.set_and_calc_cis_sub(x+dx);
	f.tail(sz_rel-1) = - ( model_.val_cis_sub - val_cis_sub ) / dx - dE_mpt(x);
	
	return f;
}

mat TwoPara::dc(uword const& sz) {
	double dx = 1e-3;
	mat coef_ = join_d<double>(vec{1}, vec_cis_sub.head_cols(sz-1));

	TwoPara model_(E_mpt, E_fil, bath, cpl, n_occ);
	model_.set_and_calc_cis_sub(x+dx);
	mat coef = join_d<double>(vec{1}, model_.vec_cis_sub.head_cols(sz-1));

	adj_phase(coef_, coef);

	mat overlap = coef_.t() * ovl(vec_do, vec_o, vec_dv, vec_v, model_.vec_do, model_.vec_o, model_.vec_dv, model_.vec_v) * coef;

	// Lowdin-orthoginalization
	overlap *= inv_sympd( sqrtmat_sympd( overlap.t() * overlap ) );

	// derivative coupling matrix
	return real( logmat(overlap) ) / dx;
}


