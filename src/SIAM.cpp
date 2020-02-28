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

mat SIAM::Pdodv() {
	return mat{vec_do(0) * vec_dv(0)};
}

mat SIAM::Pdv() {
	return mat{vec_dv(0) * vec_dv(0)};
}

mat SIAM::Pdo() {
	return mat{vec_do(0) * vec_do(0)};
}

mat SIAM::Pdob() {
	return vec_do(0) * vec_v.row(0);
}

mat SIAM::Pjdv() {
	return vec_dv(0) * vec_o.row(0);
}

mat SIAM::Pdvb() {
	return vec_dv(0) * vec_v.row(0);
}

mat SIAM::Pjdo() {
	return vec_do(0) * vec_o.row(0);
}

mat SIAM::Pab() {
    return vec_v.row(0).t() * vec_v.row(0);
}

mat SIAM::Pji() {
    return vec_o.row(0).t() * vec_o.row(0);
}

mat SIAM::Fdo() {
	return vec_do.t() * F() * vec_do;
}

mat SIAM::Fdv() {
	return vec_dv.t() * F() * vec_dv;
}

mat SIAM::Fo() {
	return vec_o.t() * F() * vec_o;
}

mat SIAM::Fv() {
	return vec_v.t() * F() * vec_v;
}

mat SIAM::Fdvb() {
	return vec_dv.t() * F() * vec_v;
}

mat SIAM::Fjdo() {
	return -vec_do.t() * F() * vec_o;
}

mat SIAM::H_gnd_gnd() {
	return mat{E_mf};
}

mat SIAM::H_gnd_dodv() {
	return mat{0};
}

mat SIAM::H_gnd_dob() {
	return zeros(1, n_vir-1);
}

mat SIAM::H_gnd_jdv() {
	return zeros(1, n_occ-1);
}

mat SIAM::H_gnd_ovov() {
	return U * Pdodv() * Pdodv();
}

mat SIAM::H_gnd_ovob() {
   	return sqrt(2) * U * Pdodv() * Pdob();
}

mat SIAM::H_gnd_ovjv() {
    return sqrt(2) * U * Pdodv() * Pjdv();
}

mat SIAM::H_dodv_dodv() {
	return E_mf + Fdv() - Fdo() + U * Pdodv() * Pdodv();
}

mat SIAM::H_dodv_dob() {
    return Fdvb() + U * Pdodv() * Pdob();
}

mat SIAM::H_dodv_jdv() {
	return -Fjdo() + U * Pdodv() * Pjdv();
}

mat SIAM::H_dodv_ovov() {
   	return sqrt(2.0) * U * Pdodv() * (Pdv()-Pdo());
}

mat SIAM::H_dodv_ovob() {
	return U * Pdodv() * (Pdv()-Pdo()) * Pdob() + U * Pdodv() * Pdvb();
}

mat SIAM::H_dodv_ovjv() {
   	return U * Pdodv() * (Pdv()-Pdo()) * Pjdv() - U * Pdodv() * Pjdo();
}

mat SIAM::H_doa_dob() {
	return Iv*(E0-Fdo) + Fv + U * (Pdob'*Pdob);
}

void SIAM::solve_cisnd() {
    
    H_doa_jdv = U * Pdob' * Pjdv;
    H_doa_ovov = sqrt(2) * U * Pdvb' * Pdodv;
    H_doa_ovob = U * (Pdvb'*Pdob + Pdodv*Pab - Iv*Pdodv*Pdo);
    H_doa_ovjv = U * Pdvb' * Pjdv;
    
    H_idv_jdv = Io*(E0+Fdv) - Fo + U * (Pjdv' * Pjdv);
    H_idv_ovov = -sqrt(2) * U * Pdodv * Pjdo';
    H_idv_ovob = -U * Pjdo' * Pdob;
    H_idv_ovjv = U * (-Pjdv'*Pjdo - Pdodv*Pji + Io*Pdodv*Pdv);
    
    
    H_ovov_ovov = E0 + 2*(Fdv-Fdo) + U * (Pdv-Pdo)^2;
    H_ovov_ovob = sqrt(2) * (Fdvb + U * (Pdv-Pdo) * Pdvb);
    H_ovov_ovjv = -sqrt(2) * (Fjdo + U * (Pdv-Pdo) * Pjdo);
    
    H_ovoa_ovob = Iv*(E0+Fdv-Fdo) - Iv*(Fdo+U*(Pdv-Pdo)*Pdo) + Fv + U*(Pdv-Pdo)*Pab + U*(Pdvb'*Pdvb);
    H_ovoa_ovjv = -U * Pdvb' * Pjdo;
    
    H_oviv_ovjv = Io*(E0+Fdv-Fdo) + Io*(Fdv+U*(Pdv-Pdo)*Pdv) - Fo - U*(Pdv-Pdo)*Pji + U*(Pjdo'*Pjdo);

    
    n_gnd_gnd = n_imp;
    n_gnd_dodv = Pdodv/sqrt(2);
    n_gnd_dob = Pdob/sqrt(2);
    n_gnd_jdv = Pjdv/sqrt(2);
    n_gnd_ovov = 0;
    n_gnd_ovob = sparse(1, n_vir-1);
    n_gnd_ovjv = sparse(1, n_occ-1);
    
    n_dodv_dodv = n_imp + 0.5*(Pdv-Pdo);
    n_dodv_dob = 0.5*Pdvb;
    n_dodv_jdv = -0.5*Pjdo;
    n_dodv_ovov = Pdodv / sqrt(2);
    n_dodv_ovob = 0.5*Pdob;
    n_dodv_ovjv = 0.5*Pjdv;
    
    n_doa_dob = Iv*n_imp + 0.5*(Pab-Iv*Pdo);
    n_doa_jdv = sparse(n_vir-1, n_occ-1);
    n_doa_ovov = sparse(n_vir-1, 1);
    n_doa_ovob = 0.5*Iv*Pdodv;
    n_doa_ovjv = sparse(n_vir-1, n_occ-1);
    
    n_idv_jdv = Io*n_imp + 0.5*(Io*Pdv-Pji);
    n_idv_ovov = sparse(n_occ-1, 1);
    n_idv_ovob = sparse(n_occ-1, n_vir-1);
    n_idv_ovjv = 0.5*Io*Pdodv;
    
    n_ovov_ovov = n_imp + Pdv - Pdo;
    n_ovov_ovob = Pdvb / sqrt(2);
    n_ovov_ovjv = -Pjdo / sqrt(2);
    
    n_ovoa_ovob = Iv*(n_imp-Pdo) + 0.5*(Pab+Iv*Pdv);
    n_ovoa_ovjv = sparse(n_vir-1, n_occ-1);
    
    n_oviv_ovjv = Io*(n_imp+Pdv) - 0.5*(Pji+Io*Pdo);
}

void SIAM::calc_bath() {

}

void SIAM::calc_Gamma() {

}

void SIAM::calc_dc() {

}


void SIAM::move_new_to_old() {
	_x = x;
}



