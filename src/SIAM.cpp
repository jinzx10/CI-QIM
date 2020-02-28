#include "SIAM.h"
#include "math_helper.h"

using namespace arma;

SIAM::SIAM(
		d2d Ed_,
		d2d E_nuc_,
		vec const& bath_,
		vec const& cpl_,
		double const& U_,
		uword const& n_occ_
): 
	Ed(Ed_), E_nuc(E_nuc_), bath(bath_), cpl(cpl_), U(U_), 
	n_bath(bath.n_elem), n_occ(n_occ_), n_vir(n_bath+1-n_occ),
	span_occ(span(0, n_occ-1)), span_vir(span(n_occ, n_bath))
{
	h = diagmat(join_cols(vec{0}, bath));
	h(0, span(1, n_bath)) = cpl.t();
	h(span(1, n_bath), 0) = cpl;
	n_mf = 0;

	// initialize data
	solve_mf();
}

void SIAM::set_and_calc(double const& x_) {
	move_new_to_old();
	h(0,0) = Ed(x_);
	solve_mf();
}

sp_mat SIAM::F(double const& n) {
	sp_mat Fock = h;
	Fock(0,0) += U * n;
	return Fock;
}

sp_mat SIAM::F() {
	sp_mat Fock = h;
	Fock(0,0) += U * n_mf;
	return Fock;
}

double SIAM::n2n(double const& n) {
	mat eigvec;
	vec eigval;
	eig_sym(eigval, eigvec, conv_to<mat>::from(F(n)));
	return accu(square(eigvec(0, span(0, n_occ-1))));
}

void SIAM::solve_mf() {
	auto dn = [this] (double const& n) { return n2n(n) - n; };
	newtonroot(dn, n_mf);
	eig_sym(val_mf, vec_mf, conv_to<mat>::from(F()));
	E_mf = accu(val_mf.head(n_occ)) - U * n_mf * n_mf;
}

void SIAM::rotate_orb() {
	subrotate(vec_mf.cols(span_occ), val_do, vec_do, vec_o, H_o, H_do_o);
	subrotate(vec_mf.cols(span_vir), val_dv, vec_dv, vec_v, H_v, H_dv_v);
}

void SIAM::subrotate(mat const& vec_sub, double& val_d, vec& vec_d, mat& vec_other, sp_mat& H_other, mat& H_d_other) {
	uword sz = vec_sub.n_cols;

	// first rotation: separate the Schmidt orbital from the subspace
	mat Q = eye(sz, sz);
	Q.col(0) = vec_sub.row(0).t();
	mat q, r;
	qr_econ(q, r, Q);

	vec_d = vec_sub * q.col(0);
	vec_other = vec_sub * q.tail_cols(sz-1);
	val_d = as_scalar( vec_d.t() * F() * vec_d );

	// second rotation: make H diagonal in the "other" subspace
	vec val_other;
	eig_sym( val_other, q, vec_other.t() * F() * vec_other);
	vec_other *= q;
	H_other = diagmat(val_other);
	H_d_other= vec_d.t() * F() * vec_other;
}


void SIAM::move_new_to_old() {
	_x = x;
}



