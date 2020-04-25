#include "SIAM.h"
#include "math_helper.h"
#include "arma_helper.h"
#include <cassert>

using namespace arma;

SIAM::SIAM(
		d2d						E_imp_,
		d2d 					E_nuc_,
		vec			const&		bath_,
		vec 		const& 		cpl_,
		double		const&		U_,
		uword		const& 		n_occ_,
		uword		const&		sz_sub_,
		double		const&		x0_
): 
	E_imp(E_imp_), E_nuc(E_nuc_), bath(bath_), cpl(cpl_), U(U_), 
	n_bath(bath.n_elem), n_occ(n_occ_), n_vir(n_bath+1-n_occ),
	span_occ(span(0, n_occ-1)), span_vir(span(n_occ, n_bath)),
	sz_sub(sz_sub_), span_sub(span(0, sz_sub-1))
{
	h = diagmat(join_cols(vec{E_imp(x0_)}, bath));
	h(0, span(1, n_bath)) = cpl.t();
	h(span(1, n_bath), 0) = cpl;
	n_mf = (h(0,0) < bath(n_occ));

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
	calc_dc_adi();
}

mat SIAM::F() {
	sp_mat Fock = h;
	Fock(0,0) += U * n_mf;
	return conv_to<mat>::from(Fock);
}

mat SIAM::F(double const& n) {
	sp_mat Fock = h;
	Fock(0,0) += U * n;
	return conv_to<mat>::from(Fock);
}

double SIAM::n2n(double const& n) {
	mat eigvec;
	vec eigval;
	eig_sym( eigval, eigvec, F(n) );
	return accu( square(eigvec(0, span_occ)) );
}

void SIAM::solve_mf() {
	auto dn = [this] (double const& n) { return n2n(n) - n; };
	newtonroot(dn, n_mf);
	eig_sym( val_mf, vec_mf, F() );
	E_mf = accu(val_mf(span_occ)) - U * n_mf * n_mf;
}

void SIAM::rotate_orb() {
	subrotate(vec_mf.cols(span_occ), vec_do, vec_o, F(), Fdodo, Fij);
	subrotate(vec_mf.cols(span_vir), vec_dv, vec_v, F(), Fdvdv, Fab);
}

void SIAM::adj_orb_sign() {
	zeyu_sign(_vec_do, vec_do);
	zeyu_sign(_vec_dv, vec_dv);
	zeyu_sign(_vec_o, vec_o);
	zeyu_sign(_vec_v, vec_v);
}

void SIAM::calc_basic_elem() {
	Pdodo = vec_do(0) * vec_do(0);
	Pdodv = vec_do(0) * vec_dv(0);;
	Pdvdv = vec_dv(0) * vec_dv(0);;

	Fdoj = vec_do.t() * F() * vec_o;
	Fdvb = vec_dv.t() * F() * vec_v;
}

void SIAM::solve_cisnd() {
	eig_sym( val_cisnd, vec_cisnd, H_cisnd() );
	coef = mat(vec_cisnd.memptr(), vec_cisnd.n_rows, sz_sub, false);
	n_cisnd = sum( vec_cisnd % (N_cisnd() * vec_cisnd) , 0).t();
}

mat SIAM::H_cisnd() {
	return join<mat>( {
			{ H_gnd_gnd()      , H_gnd_dodv()      , H_gnd_dob()      , H_gnd_jdv()      , H_gnd_ovov()      , H_gnd_ovob()      , H_gnd_ovjv()  }, 
			{ H_gnd_dodv().t() , H_dodv_dodv()     , H_dodv_dob()     , H_dodv_jdv()     , H_dodv_ovov()     , H_dodv_ovob()     , H_dodv_ovjv() }, 
			{ H_gnd_dob().t()  , H_dodv_dob().t()  , H_doa_dob()      , H_doa_jdv()      , H_doa_ovov()      , H_doa_ovob()      , H_doa_ovjv()  },
			{ H_gnd_jdv().t()  , H_dodv_jdv().t()  , H_doa_jdv().t()  , H_idv_jdv()      , H_idv_ovov()      , H_idv_ovob()      , H_idv_ovjv()  }, 
			{ H_gnd_ovov().t() , H_dodv_ovov().t() , H_doa_ovov().t() , H_idv_ovov().t() , H_ovov_ovov()     , H_ovov_ovob()     , H_ovov_ovjv() },
			{ H_gnd_ovob().t() , H_dodv_ovob().t() , H_doa_ovob().t() , H_idv_ovob().t() , H_ovov_ovob().t() , H_ovoa_ovob()     , H_ovoa_ovjv() },
			{ H_gnd_ovjv().t() , H_dodv_ovjv().t() , H_doa_ovjv().t() , H_idv_ovjv().t() , H_ovov_ovjv().t() , H_ovoa_ovjv().t() , H_oviv_ovjv() } 
	} );
}

mat SIAM::N_cisnd() {
	return join<mat>( {
			{ N_gnd_gnd()      , N_gnd_dodv()      , N_gnd_dob()      , N_gnd_jdv()      , N_gnd_ovov()      , N_gnd_ovob()      , N_gnd_ovjv()  }, 
			{ N_gnd_dodv().t() , N_dodv_dodv()     , N_dodv_dob()     , N_dodv_jdv()     , N_dodv_ovov()     , N_dodv_ovob()     , N_dodv_ovjv() }, 
			{ N_gnd_dob().t()  , N_dodv_dob().t()  , N_doa_dob()      , N_doa_jdv()      , N_doa_ovov()      , N_doa_ovob()      , N_doa_ovjv()  },
			{ N_gnd_jdv().t()  , N_dodv_jdv().t()  , N_doa_jdv().t()  , N_idv_jdv()      , N_idv_ovov()      , N_idv_ovob()      , N_idv_ovjv()  }, 
			{ N_gnd_ovov().t() , N_dodv_ovov().t() , N_doa_ovov().t() , N_idv_ovov().t() , N_ovov_ovov()     , N_ovov_ovob()     , N_ovov_ovjv() },
			{ N_gnd_ovob().t() , N_dodv_ovob().t() , N_doa_ovob().t() , N_idv_ovob().t() , N_ovov_ovob().t() , N_ovoa_ovob()     , N_ovoa_ovjv() },
			{ N_gnd_ovjv().t() , N_dodv_ovjv().t() , N_doa_ovjv().t() , N_idv_ovjv().t() , N_ovov_ovjv().t() , N_ovoa_ovjv().t() , N_oviv_ovjv() } 
	} );
}


void SIAM::calc_bath() {

}

void SIAM::calc_Gamma() {

}

void SIAM::calc_dc_adi() {
	mat S = S_exact(_vec_do, _vec_o, _vec_dv, _vec_v, vec_do, vec_o, vec_dv, vec_v);
	zeyu_sign(_coef, coef, S);
	dc_adi = calc_dc(_coef, coef, x-_x, S);
}


void SIAM::move_new_to_old() {
	_x = x;
	_vec_do = vec_do;
	_vec_dv = vec_dv;
	_vec_o = vec_o;
	_vec_v = vec_v;
	_val_cisnd = val_cisnd;
	_coef = coef;
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

mat calc_dc(mat const& _coef, mat const& coef, double const& dx, mat const& S) {
	mat overlap;
	if (S.is_empty())
		overlap = _coef.t() * coef;
	else
		overlap = _coef.t() * S * coef;

	// Lowdin-orthoginalization
	mat sqrtm_oto = sqrtmat_sympd( overlap.t() * overlap );
	if (eig_sym(sqrtm_oto)(0) < 0)
		sqrtm_oto *= -1;
	overlap *= inv_sympd(sqrtm_oto);
	//overlap *= inv_sympd( sqrtmat_sympd( overlap.t() * overlap ) );

	return real( logmat(overlap) ) / dx;
}

mat S_exact(vec const& _vec_do, mat const& _vec_o, vec const& _vec_dv, mat const& _vec_v, vec const& vec_do, mat const& vec_o, vec const& vec_dv, mat const& vec_v) {
	uword n_occ = vec_o.n_cols + 1;
	uword n_vir = vec_v.n_cols + 1;
	uvec occ = range(0, n_occ-1);
	uvec vir = range(n_occ, n_occ+n_vir-1);
	uvec i = range(1, n_occ-1);
	uvec j = range(1, n_occ-1);
	uvec d0 = uvec{0};
	uvec dv = uvec{n_occ};

	mat ovl = join_r(mat{_vec_do}, _vec_o, mat{_vec_dv}, _vec_v).t() * 
		join_r(mat{vec_do}, vec_o, mat{vec_dv}, vec_v);

	///////////////////////////////////////////////////////////
	//							M
	///////////////////////////////////////////////////////////
	// Mab
	uvec p = cat(d0, vir);
	uvec q = cat(d0, vir);
	mat M1 = det(ovl(i,j)) * ( ovl(p,q) - ovl(p,j) * solve(ovl(i,j), ovl(i,p)) );

	// Mij
	p = cat(dv, i);
	q = cat(dv, j);
	mat Y = ovl(p,q);
	Y.row(0) *= -1;
	Y.col(0) *= -1;
	mat M2 = det(Y) * inv(Y).t();

	// Maj
	q = cat(dv, occ);
	mat Z = ovl(i,q);
	mat ns = null(Z);
	mat Ro = ovl(d0,q) * ns;
	mat Ra = ovl(vir,q) * ns;
	mat u = Ra.col(1) / ( Ra.col(1)*Ro(0) - Ra.col(0)*Ro(1) );
	mat v = 1.0/Ro(1) - Ro(0)/Ro(1)*u;
	mat x = ns * join_cols(u.t(), v.t());
	mat M3 = x.tail_rows(x.n_rows-2).t().eval().each_col() % 
		( det(ovl(occ,occ)) * ( ovl(vir,dv) - ovl(vir,occ) * solve(ovl(occ,occ), ovl(occ,dv)) ) );

	// Mib
	mat ovl2 = ovl.t();
	q = cat(dv, occ);
	Z = ovl2(i,q);
	ns = null(Z);
	Ro = ovl2(d0,q) * ns;
	Ra = ovl2(vir,q) * ns;
	u = Ra.col(1) / ( Ra.col(1)*Ro(0) - Ra.col(0)*Ro(1) );
	v = 1.0/Ro(1) - Ro(0)/Ro(1)*u;
	x = ns * join_cols(u.t(), v.t());
	mat M4 = x.tail_rows(x.n_rows-2).t().eval().each_col() % 
		( det(ovl2(occ,occ)) * ( ovl2(vir,dv) - ovl2(vir,occ) * solve(ovl2(occ,occ), ovl2(occ,dv)) ) );
	M4 = M4.t();

	// individual M block
	span r = span(2, M1.n_rows-1);
	span c = span(2, M1.n_cols-1);
	double M00 = M1(0,0);
	double M01 = M1(0,1);
	mat M0b = M1(0, c);
	double M10 = M1(1,0);
	double M11 = M1(1,1);
	mat M1b = M1(1, c);
	mat Ma0 = M1(r, 0);
	mat Ma1 = M1(r, 1);
	mat Mab = M1(r, c);

	r = span(1, M2.n_rows-1);
	c = span(1, M2.n_cols-1);
	mat M0j = M2(0, c);
	mat Mi0 = M2(r, 0);
	mat Mij = M2(r, c);

	mat M1j = M3.row(0);
	mat Maj = M3.tail_rows(M3.n_rows-1);

	mat Mi1 = M4.col(0);
	mat Mib = M4.tail_cols(M4.n_cols-1);


	mat SA = join<mat>({
			{mat{M00*M00}			, mat{sqrt(2.0)*M00*M01}, sqrt(2.0)*M00*M0b	, sqrt(2.0)*M00*M0j}, 
			{mat{sqrt(2.0)*M00*M10}	, mat{M00*M11+M10*M01}	, M00*M1b+M10*M0b	, M00*M1j+M10*M0j},
			{sqrt(2.0)*M00*Ma0		, M00*Ma1+Ma0*M01		, M00*Mab+Ma0*M0b	, M00*Maj+Ma0*M0j},
			{sqrt(2.0)*M00*Mi0		, M00*Mi1+Mi0*M01		, M00*Mib+Mi0*M0b	, M00*Mij+Mi0*M0j} 
	});

	mat SB = join<mat>({
			{mat{M10*M10}			, sqrt(2.0)*M10*M0b	, sqrt(2.0)*M01*M0j},
			{mat{sqrt(2.0)*M01*M11}	, M11*M0b+M01*M1b	, M11*M0j+M01*M1j},
			{sqrt(2.0)*M01*Ma1		, Ma1*M0b+M01*Mab	, Ma1*M0j+M01*Maj},
			{sqrt(2.0)*M01*Mi1		, Mi1*M0b+M01*Mib	, Mi1*M0j+M01*Mij}
	});

	mat SC = join<mat>({ 
			{mat{M10*M10}		, mat{sqrt(2.0)*M11*M10}, sqrt(2.0)*M1b*M10	, sqrt(2.0)*M1j*M10},
			{sqrt(2.0)*Ma0*M10	, Ma0*M11+Ma1*M10		, Ma0*M1b+Mab*M10	, Ma0*M1j+Maj*M10},
			{sqrt(2.0)*Mi0*M10	, Mi0*M11+Mi1*M10		, Mi0*M1b+Mib*M10	, Mi0*M1j+Mij*M10}
	});

	mat SD = join<mat>({
			{mat{M11*M11}		, sqrt(2.0)*M11*M1b	, sqrt(2.0)*M11*M1j},
			{sqrt(2.0)*M11*Ma1	, M11*Mab+Ma1*M1b	, M11*Maj+Ma1*M1j},
			{sqrt(2.0)*M11*Mi1	, M11*Mib+Mi1*M1b	, M11*Mij+Mi1*M1j}
	});

	return join<mat>({{SA, SB}, {SC, SD}});
}

sp_mat SIAM::Io() {
	return speye(n_occ-1, n_occ-1);
}

sp_mat SIAM::Iv() {
	return speye(n_vir-1, n_vir-1);
}

///////////////////////////////////////////////////////////
//		below are all the boring matrix elements
///////////////////////////////////////////////////////////
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
	return mat{U * Pdodv * Pdodv};
}

mat SIAM::H_gnd_ovob() {
   	return zeros(1, n_vir-1);
}

mat SIAM::H_gnd_ovjv() {
   	return zeros(1, n_occ-1);
}

mat SIAM::H_dodv_dodv() {
	return mat{E_mf + Fdvdv - Fdodo + U * Pdodo * Pdvdv};
}

mat SIAM::H_dodv_dob() {
    return Fdvb;
}

mat SIAM::H_dodv_jdv() {
	return -Fdoj;
}

mat SIAM::H_dodv_ovov() {
   	return mat{sqrt(2.0) * U * Pdodv * (Pdvdv-Pdodo)};
}

mat SIAM::H_dodv_ovob() {
	return zeros(1, n_vir-1);
}

mat SIAM::H_dodv_ovjv() {
	return zeros(1, n_occ-1);
}

mat SIAM::H_doa_dob() {
	return conv_to<mat>::from(Iv()*(E_mf-Fdodo) + Fab);
}

mat SIAM::H_doa_jdv() {
	return zeros(n_vir-1, n_occ-1);
}

mat SIAM::H_doa_ovov() {
	return zeros(n_vir-1, 1);
}

mat SIAM::H_doa_ovob() {
	return conv_to<mat>::from(-U * Iv() * Pdodv * Pdodo);
}

mat SIAM::H_doa_ovjv() {
	return zeros(n_vir-1, n_occ-1);
}

mat SIAM::H_idv_jdv() {
   	return conv_to<mat>::from(Io()*(E_mf+Fdvdv) - Fij);
}

mat SIAM::H_idv_ovov() {
	return zeros(n_occ-1, 1);
}

mat SIAM::H_idv_ovob() {
	return zeros(n_occ-1, n_vir-1);
}

mat SIAM::H_idv_ovjv() { 
	return conv_to<mat>::from(U * Io() * Pdodv * Pdvdv);
}

mat SIAM::H_ovov_ovov() {
	return mat{ E_mf + 2.0*(Fdvdv-Fdodo) + 
		U * (Pdvdv-Pdodo) * (Pdvdv-Pdodo) };
}

mat SIAM::H_ovov_ovob() {
   	return sqrt(2.0) * Fdvb;
}

mat SIAM::H_ovov_ovjv() {
	return -sqrt(2.0) * Fdoj;
}

mat SIAM::H_ovoa_ovob() {
	return conv_to<mat>::from( Iv() * ( E_mf + Fdvdv - Fdodo ) - 
		Iv() * ( Fdodo + U * (Pdvdv-Pdodo) * Pdodo) + Fab );
}

mat SIAM::H_ovoa_ovjv() {
   	return zeros(n_vir-1, n_occ-1);
}

mat SIAM::H_oviv_ovjv() {
	return conv_to<mat>::from( Io() * ( E_mf + Fdvdv - Fdodo ) + 
		Io() * ( Fdvdv + U * (Pdvdv-Pdodo) * Pdvdv ) - Fij );
}

mat SIAM::N_gnd_gnd() { 
	return mat{n_mf};
}

mat SIAM::N_gnd_dodv() {
	return mat{Pdodv/sqrt(2.0)};
}

mat SIAM::N_gnd_dob() {
	return zeros(1, n_vir-1);
}

mat SIAM::N_gnd_jdv() {
	return zeros(1, n_occ-1);
}

mat SIAM::N_gnd_ovov() {
	return mat{0};
}

mat SIAM::N_gnd_ovob() {
	return zeros(1, n_vir-1);
}

mat SIAM::N_gnd_ovjv() {
	return zeros(1, n_occ-1);
}

mat SIAM::N_dodv_dodv() {
	return mat{n_mf + 0.5*(Pdvdv-Pdodo)};
}
	
mat SIAM::N_dodv_dob() {
	return zeros(1, n_vir-1);
}

mat SIAM::N_dodv_jdv() {
	return zeros(1, n_occ-1);
}

mat SIAM::N_dodv_ovov() {
	return mat{Pdodv / sqrt(2.0)};
}

mat SIAM::N_dodv_ovob() {
	return zeros(1, n_vir-1);
}

mat SIAM::N_dodv_ovjv() {
	return zeros(1, n_occ-1);
}

mat SIAM::N_doa_dob() {
	return conv_to<mat>::from(Iv()*n_mf - 0.5*Iv()*Pdodo);
}

mat SIAM::N_doa_jdv() {
	return zeros(n_vir-1, n_occ-1);
}

mat SIAM::N_doa_ovov() {
	return zeros(n_vir-1, 1);
}

mat SIAM::N_doa_ovob() {
	return conv_to<mat>::from(0.5*Iv()*Pdodv);
}

mat SIAM::N_doa_ovjv() {
	return zeros(n_vir-1, n_occ-1);
}

mat SIAM::N_idv_jdv() {
	return conv_to<mat>::from(Io()*n_mf + 0.5*Io()*Pdvdv);
}

mat SIAM::N_idv_ovov() {
	return zeros(n_occ-1, 1);
}

mat SIAM::N_idv_ovob() {
	return zeros(n_occ-1, n_vir-1);
}

mat SIAM::N_idv_ovjv() {
	return conv_to<mat>::from(0.5*Io()*Pdodv);
}

mat SIAM::N_ovov_ovov() {
	return mat{n_mf + Pdvdv - Pdodo};
}

mat SIAM::N_ovov_ovob() {
	return zeros(1, n_vir-1);
}

mat SIAM::N_ovov_ovjv() {
	return zeros(1, n_occ-1);
}

mat SIAM::N_ovoa_ovob() {
	return conv_to<mat>::from(Iv()*(n_mf-Pdodo) + 0.5*Iv()*Pdvdv);
}

mat SIAM::N_ovoa_ovjv() {
	return zeros(n_vir-1, n_occ-1);
}

mat SIAM::N_oviv_ovjv() {
	return conv_to<mat>::from(Io()*(n_mf+Pdvdv) - 0.5*Io()*Pdodo);
}


