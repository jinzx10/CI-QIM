#include <cstdlib>
#include "TwoPara.h"
#include "arma_helper.h"
#include "math_helper.h"
#include "widgets.h"

using namespace arma;

TwoPara::TwoPara(
		d2d                     E_mpt_,
		d2d                     E_fil_,
		vec         const&      bath_,
		vec         const&      cpl_,
		uword       const&      n_occ_,
		uword       const&      sz_sub_,
		double      const&      x_init_
):
	E_nuc(E_mpt_), 
	E_imp([=] (double const& x_) { return E_fil_(x_) - E_mpt_(x_); }),
	bath(bath_), cpl(cpl_), 
	n_occ(n_occ_), sz_sub(sz_sub_), x(x_init_)
{
	dE_nuc= grad(E_nuc);
	dE_imp = grad(E_imp);

	n_bath = bath.n_elem;
	dE_bath_avg = ( bath.max() - bath.min() ) / (bath.n_elem - 1);

	n_vir = n_bath + 1 - n_occ;
	span_occ = span(0, n_occ-1);
	span_vir = span(n_occ, n_bath);
	sz_cis = n_occ + n_vir - 1; // size of the selected CIS subspace

	H = diagmat(join_cols(vec{0}, bath));
	H(span(1, n_bath), 0) = cpl;
	H(0, span(1, n_bath)) = cpl.t();

	Gamma_rlx = zeros(sz_sub);

	// initialize data
	solve_orb();
	rotate_orb();
	calc_basic_elem();
	solve_slt_cis();
}

void TwoPara::set_and_calc(double const& x_) {
	move_new_to_old();
	x = x_;
	solve_orb();
	rotate_orb();
	adj_orb_sign();
	calc_basic_elem();

	solve_slt_cis();

	calc_force();
	calc_Gamma_rlx();
	calc_dc_adi();
}

void TwoPara::solve_orb() {
	H(0,0) = E_imp(x);
	eig_sym( val_H, vec_H, H );
	ev_n = accu( square( vec_H(0, span_occ) ) );
	ev_H = accu( val_H(span_occ) );
}

void TwoPara::rotate_orb() {
	subrotate(vec_H.cols(span_occ), vec_do, vec_o, H, H_dodo, H_ij);
	subrotate(vec_H.cols(span_vir), vec_dv, vec_v, H, H_dvdv, H_ab);
}

void TwoPara::adj_orb_sign() {
	zeyu_sign(_vec_do, vec_do);
	zeyu_sign(_vec_dv, vec_dv);
	zeyu_sign(_vec_o, vec_o);
	zeyu_sign(_vec_v, vec_v);
}

void TwoPara::calc_basic_elem() {
	H_doj = vec_do.t() * H * vec_o;
	H_dvb = vec_dv.t() * H * vec_v;
}

void TwoPara::solve_slt_cis() {
	sp_mat H_dov_dov = conv_to<sp_mat>::from( vec{ev_H - H_dodo + H_dvdv} );
	sp_mat H_dov_dob = conv_to<sp_mat>::from( H_dvb );
	sp_mat H_dov_jdv = conv_to<sp_mat>::from( -H_doj );
	sp_mat H_doa_dob = speye(n_vir-1, n_vir-1) * (ev_H - H_dodo) + H_ab;
	sp_mat H_doa_jdv = zeros<sp_mat>(n_vir-1, n_occ-1);
	sp_mat H_idv_jdv = speye(n_occ-1, n_occ-1) * (ev_H + H_dvdv) - H_ij;

	sp_mat H_slt_cis = join<sp_mat>( {
			{ H_dov_dov,     H_dov_dob,     H_dov_jdv},
			{ H_dov_dob.t(), H_doa_dob,     H_doa_jdv},
			{ H_dov_jdv.t(), H_doa_jdv.t(), H_idv_jdv} } );

	eigs_sym( val_slt_cis, vec_slt_cis, H_slt_cis, sz_sub-1, "sa");
}

vec TwoPara::E_bath() {
	return vectorise( ev_H - repmat(H_ij.diag(), 1, n_vir-1) + 
			repmat( H_ab.diag().t(), n_occ-1, 1) );
}

void TwoPara::calc_Gamma_rlx() {
	mat V_adi = vec_slt_cis.t() * join<sp_mat>( {
			{ sp_mat( 1, (n_occ-1)*(n_vir-1) ) },
			{ -kron( speye(n_vir-1, n_vir-1), sp_mat(H_doj) ) },
			{ kron( sp_mat(H_dvb), speye(n_occ-1, n_occ-1) ) }
	});
	mat delta = gauss( val_slt_cis, E_bath().as_row(), 5.0*dE_bath_avg );
	Gamma_rlx.tail(sz_sub-1) = 2.0 * datum::pi * sum( square(V_adi) % delta, 1 );
}

vec TwoPara::E_sub() {
	return join_cols( vec{ev_H}, val_slt_cis ) + E_nuc(x);
}

vec TwoPara::F_sub() {
	return join_cols( vec{F_gnd}, F_slt_cis ) - dE_nuc(x);;
}

void TwoPara::calc_force() {
	F_gnd = - dE_imp(x) * ev_n;
	F_slt_cis = - (val_slt_cis - _val_slt_cis) / (x - _x);
}

void TwoPara::calc_dc_adi() {
	mat S = S_exact(_vec_do, _vec_o, _vec_dv, _vec_v, 
			vec_do, vec_o, vec_dv, vec_v);
	zeyu_sign(_vec_slt_cis, vec_slt_cis, S(span(1, sz_cis), span(1, sz_cis)));
	mat _coef = join_d(vec{1.0}, _vec_slt_cis);
	mat coef = join_d(vec{1.0}, vec_slt_cis);
	ovl_sub_raw = _coef.t() * S * coef;
	dc_adi = real( logmat( orth_lowdin( ovl_sub_raw ) ) ) / (x - _x);
}

void TwoPara::move_new_to_old() {
	_x = x;
	_val_slt_cis = std::move(val_slt_cis);
	_vec_slt_cis = std::move(vec_slt_cis);
	_vec_do = std::move(vec_do);
	_vec_dv = std::move(vec_dv);
	_vec_o = std::move(vec_o);
	_vec_v = std::move(vec_v);
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
	uvec a = range(n_occ+1, n_occ+n_vir-1);

	uvec d0 = uvec{0};
	uvec dv = uvec{n_occ};

	mat ovl = join_r(mat{_vec_do}, _vec_o, mat{_vec_dv}, _vec_v).t() * 
		join_r(mat{vec_do}, vec_o, mat{vec_dv}, vec_v);

	///////////////////////////////////////////////////////////
	//                          M
	///////////////////////////////////////////////////////////
	// M1
	mat M1 = det(ovl(i,i)) * ( ovl(cat(d0, vir), cat(d0, vir)) - 
			ovl(cat(d0, vir),i) * solve(ovl(i,i), ovl(i, cat(d0, vir))) );

	// M2
	mat Y = ovl(cat(dv, occ), cat(dv, occ));
	Y.row(0) *= -1;
	Y.col(0) *= -1;
	mat M2 = det(Y) * inv(Y).t();
	Y.clear();

	// Maj
	mat ns;
	bool status = null_qr(ns, ovl(i, cat(dv, occ)));
	if (!status) {
		std::cout << "Maj: null failed" << std::endl;
		ovl(i, cat(dv, occ)).eval().save(expand_leading_tilde("~/Zt.dat"), raw_ascii);
		exit(EXIT_FAILURE);
	}

	mat Ro = ovl(d0, cat(dv, occ)) * ns;
	mat Ra = ovl(a, cat(dv, occ)) * ns;
	mat x = ns * ( join_rows(Ra.col(1), -Ra.col(0)).eval().each_col() / 
			( Ra.col(1)*Ro(0) - Ra.col(0)*Ro(1) )  ).t();

	mat Maj = det(ovl(occ,occ)) * ( 
			x.tail_rows(x.n_rows-2).t().eval().each_col() % 
			( ovl(a,dv) - ovl(a,occ) * solve(ovl(occ,occ), ovl(occ,dv)) ) );

	// Mib
	inplace_trans(ovl);
	status = null_qr(ns, ovl(i, cat(dv, occ)));
	if (!status) {
		std::cout << "Mib: null failed" << std::endl;
		ovl(i, cat(dv, occ)).eval().save(expand_leading_tilde("~/Zt.dat"), raw_ascii);
		exit(EXIT_FAILURE);
	}

	Ro = ovl(d0, cat(dv, occ)) * ns;
	Ra = ovl(a, cat(dv, occ)) * ns;
	x = ns * ( join_rows(Ra.col(1), -Ra.col(0)).eval().each_col() / 
			( Ra.col(1)*Ro(0) - Ra.col(0)*Ro(1) )  ).t();
	mat Mib = det(ovl(occ,occ)) * (
			x.tail_rows(x.n_rows-2).t().eval().each_col() % 
			( ovl(a,dv) - ovl(a,occ) * solve(ovl(occ,occ), ovl(occ,dv)) ) );
	inplace_trans(Mib);

	// individual M block
	span r = span(2, M2.n_rows-1);
	span c = span(2, M2.n_cols-1);

	mat SB = join_cols(M2(span(0,1), c), Maj);
	mat SC = join_rows(M2(r, span(0,1)), Mib);

	return join<mat>({{M1, SB}, {SC, M2(r,c)}});
}

