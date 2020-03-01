#include "SIAM.h"
#include "math_helper.h"
#include "arma_helper.h"

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

void SIAM::solve_cisnd() {
    
}

mat SIAM::H_cisnd() {
        H_sCISND = @() [
            H_gnd_gnd   , H_gnd_dodv   , H_gnd_dob   , H_gnd_jdv   , H_gnd_ovov   , H_gnd_ovob   , H_gnd_ovjv  ;
            H_gnd_dodv' , H_dodv_dodv  , H_dodv_dob  , H_dodv_jdv  , H_dodv_ovov  , H_dodv_ovob  , H_dodv_ovjv ;
            H_gnd_dob'  , H_dodv_dob'  , H_doa_dob   , H_doa_jdv   , H_doa_ovov   , H_doa_ovob   , H_doa_ovjv  ;
            H_gnd_jdv'  , H_dodv_jdv'  , H_doa_jdv'  , H_idv_jdv   , H_idv_ovov   , H_idv_ovob   , H_idv_ovjv  ;
            H_gnd_ovov' , H_dodv_ovov' , H_doa_ovov' , H_idv_ovov' , H_ovov_ovov  , H_ovov_ovob  , H_ovov_ovjv ;
            H_gnd_ovob' , H_dodv_ovob' , H_doa_ovob' , H_idv_ovob' , H_ovov_ovob' , H_ovoa_ovob  , H_ovoa_ovjv ;
            H_gnd_ovjv' , H_dodv_ovjv' , H_doa_ovjv' , H_idv_ovjv' , H_ovov_ovjv' , H_ovoa_ovjv' , H_oviv_ovjv  ];
        
	return join({});
}

mat SIAM::n_cisnd() {

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

// matrix elements
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

mat SIAM::Io() {
	return eye(n_occ-1, n_occ-1);
}

mat SIAM::Iv() {
	return eye(n_vir-1, n_vir-1);
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
	return Iv()*(E_mf-Fdo()) + Fv() + U * (Pdob().t()*Pdob());
}

mat SIAM::H_doa_jdv() {
	return U * Pdob().t() * Pjdv();
}

mat SIAM::H_doa_ovov() {
   	return sqrt(2.0) * U * Pdvb().t() * Pdodv();
}

mat SIAM::H_doa_ovob() {
	return U * (Pdvb().t()*Pdob() + Pdodv()*Pab() - Iv()*Pdodv()*Pdo());
}

mat SIAM::H_doa_ovjv() {
   	return U * Pdvb().t() * Pjdv();
}

mat SIAM::H_idv_jdv() {
   	return Io()*(E_mf+Fdv()) - Fo() + U * (Pjdv().t() * Pjdv());
}

mat SIAM::H_idv_ovov() {
   	return -sqrt(2.0) * U * Pdodv() * Pjdo().t();
}

mat SIAM::H_idv_ovob() {
   	return -U * Pjdo().t() * Pdob();
}

mat SIAM::H_idv_ovjv() { 
	return U * (-Pjdv().t()*Pjdo() - Pdodv()*Pji() + Io()*Pdodv()*Pdv());
}

mat SIAM::H_ovov_ovov() {
	return E_mf + 2.0*(Fdv()-Fdo()) + U * (Pdv()-Pdo()) * (Pdv()-Pdo());
}

mat SIAM::H_ovov_ovob() {
   	return sqrt(2.0) * (Fdvb() + U * (Pdv()-Pdo()) * Pdvb());
}

mat SIAM::H_ovov_ovjv() {
	return -sqrt(2.0) * (Fjdo() + U * (Pdv()-Pdo()) * Pjdo());
}

mat SIAM::H_ovoa_ovob() {
	return Iv()*(E_mf+Fdv()-Fdo()) - Iv()*(Fdo()+U*(Pdv()-Pdo())*Pdo()) 
		+ Fv() + U*(Pdv()-Pdo())*Pab() + U*(Pdvb().t()*Pdvb());
}

mat SIAM::H_ovoa_ovjv() {
   	return -U * Pdvb().t() * Pjdo();
}

mat SIAM::H_oviv_ovjv() {
	return Io()*(E_mf+Fdv()-Fdo()) + Io()*(Fdv()+U*(Pdv()-Pdo())*Pdv()) 
		- Fo() - U*(Pdv()-Pdo())*Pji() + U*(Pjdo().t()*Pjdo());
}

mat SIAM::n_gnd_gnd() { 
	return mat{n_mf};
}

mat SIAM::n_gnd_dodv() {
	return Pdodv()/sqrt(2.0);
}

mat SIAM::n_gnd_dob() {
	return Pdob()/sqrt(2.0);
}

mat SIAM::n_gnd_jdv() {
	return Pjdv()/sqrt(2.0);
}

mat SIAM::n_gnd_ovov() {
	return mat{0};
}

mat SIAM::n_gnd_ovob() {
	return zeros(1, n_vir-1);
}

mat SIAM::n_gnd_ovjv() {
	return zeros(1, n_occ-1);
}

mat SIAM::n_dodv_dodv() {
	return n_mf + 0.5*(Pdv()-Pdo());
}
	
mat SIAM::n_dodv_dob() {
	return 0.5*Pdvb();
}

mat SIAM::n_dodv_jdv() {
	return -0.5*Pjdo();
}

mat SIAM::n_dodv_ovov() {
	return Pdodv() / sqrt(2.0);
}

mat SIAM::n_dodv_ovob() {
	return 0.5*Pdob();
}

mat SIAM::n_dodv_ovjv() {
	return 0.5*Pjdv();
}

mat SIAM::n_doa_dob() {
	return Iv()*n_mf + 0.5*(Pab()-Iv()*Pdo());
}

mat SIAM::n_doa_jdv() {
	return zeros(n_vir-1, n_occ-1);
}

mat SIAM::n_doa_ovov() {
	return zeros(n_vir-1, 1);
}

mat SIAM::n_doa_ovob() {
	return 0.5*Iv()*Pdodv();
}

mat SIAM::n_doa_ovjv() {
	return zeros(n_vir-1, n_occ-1);
}

mat SIAM::n_idv_jdv() {
	return Io()*n_mf + 0.5*(Io()*Pdv()-Pji());
}

mat SIAM::n_idv_ovov() {
	return zeros(n_occ-1, 1);
}

mat SIAM::n_idv_ovob() {
	return zeros(n_occ-1, n_vir-1);
}

mat SIAM::n_idv_ovjv() {
	return 0.5*Io()*Pdodv();
}

mat SIAM::n_ovov_ovov() {
	return n_mf + Pdv() - Pdo();
}

mat SIAM::n_ovov_ovob() {
	return Pdvb() / sqrt(2.0);
}

mat SIAM::n_ovov_ovjv() {
	return -Pjdo() / sqrt(2.0);
}

mat SIAM::n_ovoa_ovob() {
	return Iv()*(n_mf-Pdo()) + 0.5*(Pab()+Iv()*Pdv());
}

mat SIAM::n_ovoa_ovjv() {
	return zeros(n_vir-1, n_occ-1);
}

mat SIAM::n_oviv_ovjv() {
	return Io()*(n_mf+Pdv()) - 0.5*(Pji()+Io()*Pdo());
}


