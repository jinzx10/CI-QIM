#include "SIAM.h"
#include "math_helper.h"
#include "arma_helper.h"

using namespace arma;

SIAM::SIAM(
		d2d                     E_imp_,
		d2d                     E_nuc_,
		vec         const&      bath_,
		vec         const&      cpl_,
		double      const&      U_,
		uword       const&      n_occ_,
		uword       const&      sz_sub_,
		double      const&      x0_
): 
	x(x0_), E_imp(E_imp_), E_nuc(E_nuc_), bath(bath_), cpl(cpl_), U(U_), 
	n_bath(bath.n_elem), n_occ(n_occ_), n_vir(n_bath+1-n_occ),
	span_occ(span(0, n_occ-1)), span_vir(span(n_occ, n_bath)),
	sz_sub(sz_sub_), span_sub(span(0, sz_sub-1)), 
	sz_cisnd(2*(n_occ+n_vir)-1), sz_jb((n_occ-1)*(n_vir-1)),
	dE_avg((bath.max()-bath.min())/n_bath)
{
	h = diagmat(join_cols(vec{E_imp(x)}, bath));
	h(0, span(1, n_bath)) = cpl.t();
	h(span(1, n_bath), 0) = cpl;
	n_mf = (h(0,0) < bath(n_occ));
	Gamma_rlx.zeros(sz_sub);

	// initialization
	solve_mf();
	rotate_orb();
	calc_basic_elem();
	solve_cisnd();
}

void SIAM::set_and_calc(double const& x_) {
	move_new_to_old();
	x = x_;
	h(0,0) = E_imp(x);

	solve_mf();

	rotate_orb();
	adj_orb_sign();
	calc_basic_elem();

	solve_cisnd();

	calc_force();
	calc_Gamma_rlx();
	calc_dc_adi();
}

sp_mat SIAM::F() {
	return F(n_mf);
}

sp_mat SIAM::F(double const& n) {
	sp_mat Fock = h;
	Fock(0,0) += U * n;
	return Fock;
}

void SIAM::solve_mf() {
	auto dn = [this] (double const& n) -> double { 
		mat eigvec;
		vec eigval;
		eig_sym( eigval, eigvec, conv_to<mat>::from(F(n)) );
		return accu( square(eigvec(0, span_occ)) ) - n;
	};
	broydenroot(dn, n_mf, 0.3, 1e-8, 200);
	eig_sym( val_mf, vec_mf, conv_to<mat>::from(F()) );
	E_mf = 2.0 * accu(val_mf(span_occ)) - U * n_mf * n_mf;
}

void SIAM::rotate_orb() {
	subrotate(vec_mf.cols(span_occ), vec_do, vec_o, conv_to<mat>::from(F()), 
			F_dodo, F_ij);
	subrotate(vec_mf.cols(span_vir), vec_dv, vec_v, conv_to<mat>::from(F()), 
			F_dvdv, F_ab);
}

void SIAM::adj_orb_sign() {
	zeyu_sign(_vec_do, vec_do);
	zeyu_sign(_vec_dv, vec_dv);
	zeyu_sign(_vec_o, vec_o);
	zeyu_sign(_vec_v, vec_v);
}

void SIAM::calc_basic_elem() {
	P_dodo = vec_do(0) * vec_do(0);
	P_dodv = vec_do(0) * vec_dv(0);;
	P_dvdv = vec_dv(0) * vec_dv(0);;

	F_doj = vec_do.t() * F() * vec_o;
	F_dvb = vec_dv.t() * F() * vec_v;
}

void SIAM::solve_cisnd() {
	eig_sym( val_cisnd, vec_cisnd, conv_to<mat>::from(H_cisnd()) );
	vec_cisnd = vec_cisnd.head_cols(sz_sub);
	val_cisnd = val_cisnd.head_rows(sz_sub);
	//eigs_sym( val_cisnd, vec_cisnd, H_cisnd(), sz_sub, "sa" );
	n_cisnd = sum( vec_cisnd % (N_cisnd() * vec_cisnd) , 0).t();
}

sp_mat SIAM::H_cisnd() {
	auto H11 = [this] () -> sp_mat {
		return join_cols(
				join_rows(H_gnd_gnd()     , H_gnd_dodv()    , H_gnd_dob()    , H_gnd_jdv()),
				join_rows(H_gnd_dodv().t(), H_dodv_dodv()   , H_dodv_dob()   , H_dodv_jdv()),
				join_rows(H_gnd_dob().t() , H_dodv_dob().t(), H_doa_dob()    , H_doa_jdv()),
				join_rows(H_gnd_jdv().t() , H_dodv_jdv().t(), H_doa_jdv().t(), H_idv_jdv())
		);
	};

	auto H21 = [this] () -> sp_mat {
		return join_cols(
				join_rows(H_gnd_ovov().t(), H_dodv_ovov().t(), H_doa_ovov().t(), H_idv_ovov().t()),
				join_rows(H_gnd_ovob().t(), H_dodv_ovob().t(), H_doa_ovob().t(), H_idv_ovob().t()),
				join_rows(H_gnd_ovjv().t(), H_dodv_ovjv().t(), H_doa_ovjv().t(), H_idv_ovjv().t())
		);
	};

	auto H12 = [this] () -> sp_mat {
		return join_cols(
				join_rows(H_gnd_ovov() , H_gnd_ovob() , H_gnd_ovjv()),
				join_rows(H_dodv_ovov(), H_dodv_ovob(), H_dodv_ovjv()),
				join_rows(H_doa_ovov() , H_doa_ovob() , H_doa_ovjv()),
				join_rows(H_idv_ovov() , H_idv_ovob() , H_idv_ovjv())
		);
	};

	auto H22 = [this] () -> sp_mat {
		return join_cols(
				join_rows(H_ovov_ovov()    , H_ovov_ovob()    , H_ovov_ovjv()),
				join_rows(H_ovov_ovob().t(), H_ovoa_ovob()    , H_ovoa_ovjv()),
				join_rows(H_ovov_ovjv().t(), H_ovoa_ovjv().t(), H_oviv_ovjv())
		);
	};

	return join_cols(join_rows(H11(), H12()), join_rows(H21(), H22()));
}

sp_mat SIAM::N_cisnd() {
	auto N11 = [this] () -> sp_mat {
		return join_cols(
				join_rows(N_gnd_gnd()     , N_gnd_dodv()    , N_gnd_dob()    , N_gnd_jdv()),
				join_rows(N_gnd_dodv().t(), N_dodv_dodv()   , N_dodv_dob()   , N_dodv_jdv()),
				join_rows(N_gnd_dob().t() , N_dodv_dob().t(), N_doa_dob()    , N_doa_jdv()),
				join_rows(N_gnd_jdv().t() , N_dodv_jdv().t(), N_doa_jdv().t(), N_idv_jdv())
		);
	};

	auto N21 = [this] () -> sp_mat {
		return join_cols(
				join_rows(N_gnd_ovov().t(), N_dodv_ovov().t(), N_doa_ovov().t(), N_idv_ovov().t()),
				join_rows(N_gnd_ovob().t(), N_dodv_ovob().t(), N_doa_ovob().t(), N_idv_ovob().t()),
				join_rows(N_gnd_ovjv().t(), N_dodv_ovjv().t(), N_doa_ovjv().t(), N_idv_ovjv().t())
		);
	};

	auto N12 = [this] () -> sp_mat {
		return join_cols(
				join_rows(N_gnd_ovov() , N_gnd_ovob() , N_gnd_ovjv()),
				join_rows(N_dodv_ovov(), N_dodv_ovob(), N_dodv_ovjv()),
				join_rows(N_doa_ovov() , N_doa_ovob() , N_doa_ovjv()),
				join_rows(N_idv_ovov() , N_idv_ovob() , N_idv_ovjv())
		);
	};

	auto N22 = [this] () -> sp_mat {
		return join_cols(
				join_rows(N_ovov_ovov()    , N_ovov_ovob()    , N_ovov_ovjv()),
				join_rows(N_ovov_ovob().t(), N_ovoa_ovob()    , N_ovoa_ovjv()),
				join_rows(N_ovov_ovjv().t(), N_ovoa_ovjv().t(), N_oviv_ovjv())
		);
	};

	return join_cols(join_rows(N11(), N12()), join_rows(N21(), N22()));
}

void SIAM::calc_force() {
	F_cisnd = -(val_cisnd - _val_cisnd) / (x - _x);
	F_nucl = -(E_nuc(x) - E_nuc(_x)) / (x - _x);
}

sp_mat SIAM::V_cpl() {
	sp_mat V(sz_cisnd, 2*sz_jb);
	V(span(2, n_vir), span(0, sz_jb-1)) = H_doa_jb();
	V(span(n_vir+1, n_vir+n_occ-1), span(0, sz_jb-1)) = H_idv_jb();
	V(span(n_occ+n_vir+1, n_occ+2*n_vir-1), span(sz_jb, 2*sz_jb-1)) 
		= H_ovoa_ovjb();
	V(span(n_occ+2*n_vir, 2*(n_occ+n_vir)-2), span(sz_jb, 2*sz_jb-1)) 
		= H_oviv_ovjb();
	return V;
}

vec SIAM::E_bath() {
	sp_mat H_ia_jb = E_mf * kron(Iv(), Io()) - kron(Iv(), F_ij) + kron(F_ab, Io());
    sp_mat H_ovia_ovjb = (E_mf + F_dvdv - F_dodo) * kron(Iv(), Io()) 
		+ kron(F_ab, Io()) - kron(Iv(), F_ij) 
		- U/2.0 * P_dvdv * P_dodo * kron(Iv(), Io());
	return vec{join_cols(H_ia_jb.diag(), H_ovia_ovjb.diag())};
}

void SIAM::calc_Gamma_rlx() {
	Gamma_rlx(0) = 0;
	for (size_t i = 1; i != sz_sub; ++i) {
		Gamma_rlx(i) = 2.0 * datum::pi * dot( 
				square( vec_cisnd.col(i).t() * V_cpl() ), 
				normpdf(E_bath(), val_cisnd(i), 5.0*dE_avg) );
	}
}

void SIAM::calc_dc_adi() {
	mat S = S_exact(_vec_do, _vec_o, _vec_dv, _vec_v, 
			vec_do, vec_o, vec_dv, vec_v);
	zeyu_sign(_vec_cisnd, vec_cisnd, S);
	ovl_sub_raw = _vec_cisnd.t() * S * vec_cisnd;
	dc_adi = real( logmat( orth_lowdin( ovl_sub_raw ) ) ) / ( x - _x );
}

void SIAM::move_new_to_old() {
	_x = x;
	_vec_do = vec_do;
	_vec_dv = vec_dv;
	_vec_o = vec_o;
	_vec_v = vec_v;
	_val_cisnd = val_cisnd;
	_vec_cisnd = vec_cisnd;
}

void subrotate(mat const& vec_sub, vec& vec_d, mat& vec_other, mat const& H, double& H_d, sp_mat& H_other) {
	uword sz = vec_sub.n_cols;

	// first rotation: separate the Schmidt orbital from the subspace
	mat Q = eye(sz, sz);
	Q.col(0) = vec_sub.row(0).t();
	mat q, r;
	qr_econ(q, r, Q);

	vec_d = vec_sub * q.col(0);
	vec_other = vec_sub * q.tail_cols(sz-1);
	H_d = as_scalar( vec_d.t() * H * vec_d );

	// second rotation: make H diagonal in the "other" subspace
	vec val_other;
	eig_sym( val_other, q, vec_other.t() * H * vec_other );
	vec_other *= q;
	H_other = diagmat(val_other);
}

void zeyu_sign(mat const& _vecs, mat& vecs, mat const& S) {
	uword sz = _vecs.n_cols;

	// crude adjustment
	for (uword i = 0; i != sz; ++i) {
		if (arma::dot(_vecs.col(i), vecs.col(i)) < 0) {
			vecs.col(i) *= -1;
		}
	}

	// Zeyu's algorithm
	mat U;
	if (S.is_empty())
		U = _vecs.t() * vecs;
	else
		U = _vecs.t() * S * vecs;

	if (det(U) < 0) {
		vecs.col(0) *= -1;
		U.col(0) *= -1;
	}

	bool is_conv = false;
	while (!is_conv) {
		is_conv = true;
		for (uword j = 0; j != sz-1; ++j) {
			for (uword k = j+1; k != sz; ++k) {
				double D = 3.0 * ( U(j,j)*U(j,j) + U(k,k)*U(k,k) ) + 
					6.0 * U(j,k) * U(k,j) + 8.0 * ( U(j,j) + U(k,k) ) -
					3.0 * ( dot(U.row(j), U.col(j)) + 
							dot(U.row(k), U.col(k)) );
				if (D < 0) {
					vecs.cols(uvec{j,k}) *= -1;
					U.cols(uvec{j,k}) *= -1;
					is_conv = false;
				}
			}
		}

	}
}

mat S_exact(vec const& _vec_do, mat const& _vec_o, vec const& _vec_dv, mat const& _vec_v, vec const& vec_do, mat const& vec_o, vec const& vec_dv, mat const& vec_v) {
	uword n_occ = vec_o.n_cols + 1;
	uword n_vir = vec_v.n_cols + 1;

	uvec occ = range(0, n_occ-1);
	uvec vir = range(n_occ, n_occ+n_vir-1);

	uvec i = range(1, n_occ-1);
	uvec j = range(1, n_occ-1);

	uvec a = range(n_occ+1, n_occ+n_vir-1);

	uvec d0 = uvec{0};
	uvec dv = uvec{n_occ};

	mat ovl = join_rows(_vec_do, _vec_o, _vec_dv, _vec_v).t() * 
		join_rows(vec_do, vec_o, vec_dv, vec_v);

	///////////////////////////////////////////////////////////
	//                          M
	///////////////////////////////////////////////////////////
	// M1
	mat M1 = det(ovl(i,i)) * ( ovl(cat(d0, vir), cat(d0, vir)) - 
			ovl(cat(d0, vir),i) * solve(ovl(i,i), ovl(i, cat(d0, vir))) );

	span r = span(2, M1.n_rows-1);
	span c = span(2, M1.n_cols-1);
	double M00 = M1(0,0);
	double M01 = M1(0,1);
	double M10 = M1(1,0);
	double M11 = M1(1,1);
	mat M0b = M1(0, c);
	mat M1b = M1(1, c);
	mat Ma0 = M1(r, 0);
	mat Ma1 = M1(r, 1);
	mat Mab = M1(r, c);
	M1.clear();

	// M2
	mat Y = ovl(cat(dv, occ), cat(dv, occ));
	Y.row(0) *= -1;
	Y.col(0) *= -1;
	mat M2 = det(Y) * inv(Y).t();
	Y.clear();

	r = span(2, M2.n_rows-1);
	c = span(2, M2.n_cols-1);
	mat M0j = M2(0, c);
	mat M1j = M2(1, c);
	mat Mi0 = M2(r, 0);
	mat Mi1 = M2(r, 1);
	mat Mij = M2(r, c);
	M2.clear();

	// Maj
	mat ns = null_qr( ovl(i, cat(dv, occ)) );
	mat Ro = ovl(d0, cat(dv, occ)) * ns;
	mat Ra = ovl(a, cat(dv, occ)) * ns;
	mat x = ns * ( join_rows(Ra.col(1), -Ra.col(0)).eval().each_col() / 
			( Ra.col(1)*Ro(0) - Ra.col(0)*Ro(1) )  ).t();

	mat Maj = det(ovl(occ,occ)) * ( 
			x.tail_rows(x.n_rows-2).t().eval().each_col() % 
			( ovl(a,dv) - ovl(a,occ) * solve(ovl(occ,occ), ovl(occ,dv)) ) );
	x.clear();

	// Mib
	ovl = ovl.t();
	ns = null_qr( ovl(i, cat(dv, occ)) );
	Ro = ovl(d0, cat(dv, occ)) * ns;
	Ra = ovl(a, cat(dv, occ)) * ns;
	x = ns * ( join_rows(Ra.col(1), -Ra.col(0)).eval().each_col() / 
			( Ra.col(1)*Ro(0) - Ra.col(0)*Ro(1) )  ).t();
	mat Mib = det(ovl(occ,occ)) * (
			x.tail_rows(x.n_rows-2).t().eval().each_col() % 
			( ovl(a,dv) - ovl(a,occ) * solve(ovl(occ,occ), ovl(occ,dv)) ) );
	Mib = Mib.t();
	x.clear();
	ns.clear();
	Ro.clear();
	Ra.clear();

	auto SA = [&] () -> mat {
		return join_cols(
				join_rows(mat{M00*M00}          , mat{sqrt(2.0)*M00*M01}, sqrt(2.0)*M00*M0b, sqrt(2.0)*M00*M0j),
				join_rows(mat{sqrt(2.0)*M00*M10}, mat{M00*M11+M10*M01}  , M00*M1b+M10*M0b  , M00*M1j+M10*M0j),
				join_rows(sqrt(2.0)*M00*Ma0     , M00*Ma1+Ma0*M01       , M00*Mab+Ma0*M0b  , M00*Maj+Ma0*M0j),
				join_rows(sqrt(2.0)*M00*Mi0     , M00*Mi1+Mi0*M01       , M00*Mib+Mi0*M0b  , M00*Mij+Mi0*M0j)
		);
	};

	auto SB = [&] () -> mat {
		return join_cols(
				join_rows(mat{M10*M10}          , sqrt(2.0)*M10*M0b, sqrt(2.0)*M01*M0j),
				join_rows(mat{sqrt(2.0)*M01*M11}, M11*M0b+M01*M1b  , M11*M0j+M01*M1j),
				join_rows(sqrt(2.0)*M01*Ma1     , Ma1*M0b+M01*Mab  , Ma1*M0j+M01*Maj),
				join_rows(sqrt(2.0)*M01*Mi1     , Mi1*M0b+M01*Mib  , Mi1*M0j+M01*Mij)
		);
	};

	auto SC = [&] () -> mat {
		return join_cols(
				join_rows(mat{M10*M10}     , mat{sqrt(2.0)*M11*M10}, sqrt(2.0)*M1b*M10, sqrt(2.0)*M1j*M10),
				join_rows(sqrt(2.0)*Ma0*M10, Ma0*M11+Ma1*M10       , Ma0*M1b+Mab*M10  , Ma0*M1j+Maj*M10),
				join_rows(sqrt(2.0)*Mi0*M10, Mi0*M11+Mi1*M10       , Mi0*M1b+Mib*M10  , Mi0*M1j+Mij*M10)
		);
	};

	auto SD = [&] () -> mat {
		return join_cols(
				join_rows(mat{M11*M11}     , sqrt(2.0)*M11*M1b, sqrt(2.0)*M11*M1j),
				join_rows(sqrt(2.0)*M11*Ma1, M11*Mab+Ma1*M1b  , M11*Maj+Ma1*M1j),
				join_rows(sqrt(2.0)*M11*Mi1, M11*Mib+Mi1*M1b  , M11*Mij+Mi1*M1j)
		);
	};

	return join_cols(join_rows(SA(), SB()), join_rows(SC(), SD()));
}

sp_mat SIAM::Io() {
	return speye(n_occ-1, n_occ-1);
}

sp_mat SIAM::Iv() {
	return speye(n_vir-1, n_vir-1);
}

///////////////////////////////////////////////////////////
//      below are all the boring matrix elements
///////////////////////////////////////////////////////////
sp_mat SIAM::H_gnd_gnd() {
	return sp_mat{mat{E_mf}};
}

sp_mat SIAM::H_gnd_dodv() {
	return sp_mat{mat{0}};
}

sp_mat SIAM::H_gnd_dob() {
	return sp_mat(1, n_vir-1);
}

sp_mat SIAM::H_gnd_jdv() {
	return sp_mat(1, n_occ-1);
}

sp_mat SIAM::H_gnd_ovov() {
	return sp_mat{mat{U * P_dodv * P_dodv}};
}

sp_mat SIAM::H_gnd_ovob() {
	return sp_mat(1, n_vir-1);
}

sp_mat SIAM::H_gnd_ovjv() {
	return sp_mat(1, n_occ-1);
}

sp_mat SIAM::H_dodv_dodv() {
	return sp_mat{mat{E_mf + F_dvdv - F_dodo + U * P_dodo * P_dvdv}};
}

sp_mat SIAM::H_dodv_dob() {
	return sp_mat{F_dvb};
}

sp_mat SIAM::H_dodv_jdv() {
	return sp_mat{-F_doj};
}

sp_mat SIAM::H_dodv_ovov() {
	return sp_mat{mat{sqrt(2.0) * U * P_dodv * (P_dvdv-P_dodo)}};
}

sp_mat SIAM::H_dodv_ovob() {
	return sp_mat(1, n_vir-1);
}

sp_mat SIAM::H_dodv_ovjv() {
	return sp_mat(1, n_occ-1);
}

sp_mat SIAM::H_doa_dob() {
	return Iv()*(E_mf-F_dodo) + F_ab;
}

sp_mat SIAM::H_doa_jdv() {
	return sp_mat(n_vir-1, n_occ-1);
}

sp_mat SIAM::H_doa_ovov() {
	return sp_mat(n_vir-1, 1);
}

sp_mat SIAM::H_doa_ovob() {
	return -U * Iv() * P_dodv * P_dodo;
}

sp_mat SIAM::H_doa_ovjv() {
	return sp_mat(n_vir-1, n_occ-1);
}

sp_mat SIAM::H_idv_jdv() {
	return Io()*(E_mf+F_dvdv) - F_ij;
}

sp_mat SIAM::H_idv_ovov() {
	return sp_mat(n_occ-1, 1);
}

sp_mat SIAM::H_idv_ovob() {
	return sp_mat(n_occ-1, n_vir-1);
}

sp_mat SIAM::H_idv_ovjv() { 
	return U * Io() * P_dodv * P_dvdv;
}

sp_mat SIAM::H_ovov_ovov() {
	return sp_mat{mat{ E_mf + 2.0*(F_dvdv-F_dodo) + 
		U * (P_dvdv-P_dodo) * (P_dvdv-P_dodo) }};
}

sp_mat SIAM::H_ovov_ovob() {
	return sp_mat{sqrt(2.0) * F_dvb};
}

sp_mat SIAM::H_ovov_ovjv() {
	return sp_mat{-sqrt(2.0) * F_doj};
}

sp_mat SIAM::H_ovoa_ovob() {
	return Iv() * ( E_mf + F_dvdv - F_dodo ) - 
		Iv() * ( F_dodo + U * (P_dvdv-P_dodo) * P_dodo) + F_ab;
}

sp_mat SIAM::H_ovoa_ovjv() {
   	return sp_mat(n_vir-1, n_occ-1);
}

sp_mat SIAM::H_oviv_ovjv() {
	return Io() * ( E_mf + F_dvdv - F_dodo ) + 
		Io() * ( F_dvdv + U * (P_dvdv-P_dodo) * P_dvdv ) - F_ij;
}

sp_mat SIAM::H_doa_jb() {
	return -kron(Iv(), sp_mat{F_doj});
}

sp_mat SIAM::H_idv_jb() {
	return kron(sp_mat{F_dvb}, Io());
}

sp_mat SIAM::H_ovoa_ovjb() {
	return -sqrt(2.0) * kron(Iv(), sp_mat{F_doj});
}

sp_mat SIAM::H_oviv_ovjb() {
	return sqrt(2) * kron(sp_mat{F_dvb}, Io());
}

sp_mat SIAM::N_gnd_gnd() { 
	return sp_mat{mat{n_mf}};
}

sp_mat SIAM::N_gnd_dodv() {
	return sp_mat{mat{P_dodv/sqrt(2.0)}};
}

sp_mat SIAM::N_gnd_dob() {
	return sp_mat(1, n_vir-1);
}

sp_mat SIAM::N_gnd_jdv() {
	return sp_mat(1, n_occ-1);
}

sp_mat SIAM::N_gnd_ovov() {
	return sp_mat{mat{0}};
}

sp_mat SIAM::N_gnd_ovob() {
	return sp_mat(1, n_vir-1);
}

sp_mat SIAM::N_gnd_ovjv() {
	return sp_mat(1, n_occ-1);
}

sp_mat SIAM::N_dodv_dodv() {
	return sp_mat{mat{n_mf + 0.5*(P_dvdv-P_dodo)}};
}

sp_mat SIAM::N_dodv_dob() {
	return sp_mat(1, n_vir-1);
}

sp_mat SIAM::N_dodv_jdv() {
	return sp_mat(1, n_occ-1);
}

sp_mat SIAM::N_dodv_ovov() {
	return sp_mat{mat{P_dodv / sqrt(2.0)}};
}

sp_mat SIAM::N_dodv_ovob() {
	return sp_mat(1, n_vir-1);
}

sp_mat SIAM::N_dodv_ovjv() {
	return sp_mat(1, n_occ-1);
}

sp_mat SIAM::N_doa_dob() {
	return Iv()*n_mf - 0.5*Iv()*P_dodo;
}

sp_mat SIAM::N_doa_jdv() {
	return sp_mat(n_vir-1, n_occ-1);
}

sp_mat SIAM::N_doa_ovov() {
	return sp_mat(n_vir-1, 1);
}

sp_mat SIAM::N_doa_ovob() {
	return 0.5*Iv()*P_dodv;
}

sp_mat SIAM::N_doa_ovjv() {
	return sp_mat(n_vir-1, n_occ-1);
}

sp_mat SIAM::N_idv_jdv() {
	return Io()*n_mf + 0.5*Io()*P_dvdv;
}

sp_mat SIAM::N_idv_ovov() {
	return sp_mat(n_occ-1, 1);
}

sp_mat SIAM::N_idv_ovob() {
	return sp_mat(n_occ-1, n_vir-1);
}

sp_mat SIAM::N_idv_ovjv() {
	return 0.5*Io()*P_dodv;
}

sp_mat SIAM::N_ovov_ovov() {
	return sp_mat{mat{n_mf + P_dvdv - P_dodo}};
}

sp_mat SIAM::N_ovov_ovob() {
	return sp_mat(1, n_vir-1);
}

sp_mat SIAM::N_ovov_ovjv() {
	return sp_mat(1, n_occ-1);
}

sp_mat SIAM::N_ovoa_ovob() {
	return Iv()*(n_mf-P_dodo) + 0.5*Iv()*P_dvdv;
}

sp_mat SIAM::N_ovoa_ovjv() {
	return sp_mat(n_vir-1, n_occ-1);
}

sp_mat SIAM::N_oviv_ovjv() {
	return Io()*(n_mf+P_dvdv) - 0.5*Io()*P_dodo;
}


