#include "TwoPara.h"
#include "arma_helper.h"
#include "math_helper.h"
#include "widgets.h"

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

mat TwoPara::dc(uword const& sz, std::string const& method) {
	double dx = 1e-4;
	mat coef_ = join_d<double>(vec{1.0}, vec_cis_sub.head_cols(sz-1));

	TwoPara model_(E_mpt, E_fil, bath, cpl, n_occ);
	model_.set_and_calc_cis_sub(x+dx);
	mat coef = join_d<double>(vec{1}, model_.vec_cis_sub.head_cols(sz-1));

	adj_phase(coef_, coef);

	mat overlap = coef_.t() * ovl(vec_do, vec_o, vec_dv, vec_v, model_.vec_do, model_.vec_o, model_.vec_dv, model_.vec_v, method) * coef;

	// Lowdin-orthoginalization
	overlap *= inv_sympd( sqrtmat_sympd( overlap.t() * overlap ) );

	// derivative coupling matrix
	return real( logmat(overlap) ) / dx;
}


// assume parallel transport
void adj_phase(mat const& vecs_old, mat& vecs_new) {
	for (uword j = 0; j != vecs_old.n_cols; ++j) {
		if ( dot(vecs_old.col(j), vecs_new.col(j)) < 0 )
			vecs_new.col(j) *= -1;
	}
}

mat ovl(vec const& vec_do_, mat const& vec_occ_, vec const& vec_dv_, mat const& vec_vir_, vec const& vec_do, mat const& vec_occ, vec const& vec_dv, mat const& vec_vir, std::string const& method) {
	mat vo_ = join_rows(vec_do_, vec_occ_);
	mat vv_ = join_rows(vec_dv_, vec_vir_);
	mat vo = join_rows(vec_do, vec_occ);
	mat vv = join_rows(vec_dv, vec_vir);

	uword n_occ = vo.n_cols;
	uword n_vir = vv.n_cols;

	// assume no trivial crossing; overlap matrix is close to identity
	adj_phase(vo_, vo);
	adj_phase(vv_, vv);

	mat ovl_orb = join_rows(vo_, vv_).t() * join_rows(vo, vv);
	mat ovl = zeros(n_occ+n_vir, n_occ+n_vir);

	// sizes and indices of the orbitals:
	//           do     occ           dv           vir 
	// size	     1     n_occ-1        1          n_vir-1
	// index     0    1 : n_occ-1    n_occ    n_occ+1 : n_occ+n_vir-1
	//
	// sizes and indices of the basis Slater determinants:
	//          Psi_{0}  Psi_{do}^{dv}   Psi_{do}^{a}     Psi_{i}^{dv}
	// size       1            1           n_vir-1          n_occ-1
	// index      0            1          2 : n_vir   n_vir+1 : n_vir+n_occ-1
	

	if ( method == "exact" ) {
		// indices for basis determinants
		span span_doa = span(1, n_vir); // dv is included in vir
		span span_idv = span(n_vir+1, n_vir+n_occ-1);

		// indices for orbitals
		span span_o = span(0, n_occ-1);
		uvec idx_occ = range(1, n_occ-1);
		span span_v = span(n_occ, n_occ+n_vir-1);
		uvec idx_v = range(n_occ, n_occ+n_vir-1);
		// first row
		mat ref = ovl_orb(span_o, span_o);
		ovl(0, 0) = det(ref);
		ovl(0, span_doa) = det12(ref, uvec{0}, ovl_orb(span_o, span_v)).t();
		ovl(0, span_idv) = det12(ref, idx_occ, ovl_orb(span_o, n_occ)).t();

		// first column
		ovl(span_doa, 0) = det12(ref, uvec{0}, ovl_orb(span_v, span_o), 'r');
		ovl(span_idv, 0) = det12(ref, idx_occ, ovl_orb(n_occ, span_o), 'r');

		// ab block
		for (uword a = 0; a != n_vir; ++a) {
			uvec idx_ref = cat(n_occ+a, range(1, n_occ-1));
			ref = ovl_orb(idx_ref, idx_ref);
			ovl(1+a, span_doa) = det12(ref, uvec{0}, ovl_orb(idx_ref, idx_v)).t();
		}

		// ij block
		for (uword i = 1; i != n_occ; ++i) {
			uvec idx_ref = cat(range(0, i-1), n_occ, range(i+1, n_occ-1));
			ref = ovl_orb(idx_ref, idx_ref);
			ovl(n_vir+i, span_idv) = -det12(ref, idx_occ, ovl_orb(idx_ref, uvec{i})).t();
			ovl(n_vir+i, n_vir+i) = det(ref);
		}

		// aj block
		for (uword a = 0; a != n_vir; ++a) {
			uvec idx_ref = cat(n_occ+a, range(1, n_occ-1));
			ref = ovl_orb(idx_ref, idx_ref);
			ovl(a+1, span_idv) = det3(ref, 0, ovl_orb(idx_ref, uvec{0}), idx_occ, ovl_orb(idx_ref, uvec{n_occ})).t();

		}

		// ib block
		for (uword b = 0; b != n_vir; ++b) {
			uvec idx_ref = cat(n_occ+b, range(1, n_occ-1));
			ref = ovl_orb(idx_ref, idx_ref);
			ovl(span_idv, b+1) = det3(ref, 0, ovl_orb(uvec{0}, idx_ref), idx_occ, ovl_orb(uvec{n_occ}, idx_ref), 'r');
		}

	} else if ( method == "dumb" ) {
		auto idx_gnd = [&] () -> uvec { return range(0, n_occ-1); };
		auto idx_doa = [&] (uword const& a) -> uvec {
			return cat(n_occ+a, range(1, n_occ-1)); // a starts from 0 (dv)
		};
		auto idx_idv = [&] (uword const& i) -> uvec {
			return cat(range(0, i-1), n_occ, range(i+1, n_occ-1));
			// i starts from 1 (the first column of occ)
		};
		auto indices = [&] (uword const& p) -> uvec {
			if (p == 0)
				return idx_gnd();
			if (p <= n_vir )
				return idx_doa(p-1);
			return idx_idv(p-n_vir);
		};

		for (uword p = 0; p != n_occ+n_vir; ++p)
			for (uword q = 0; q != n_occ+n_vir; ++q)
				ovl(p,q) = det( ovl_orb(indices(p), indices(q)) );

	} else { // approximated by orbital overlap

		/* overlap = [
		 *    1     ,  <do_|dv> , <do_|b> ,  <j_|dv>'  ;
		 * <dv_|do> ,     1     , <dv_|b> , -<j_|do>'  ;
		 * <a_|do>  ,  <a_|dv>  , <a_|b>  ,    0	   ;
		 * <dv_|i>' , -<do_|i>' ,    0    ,   (*)		];
		 *
		 * (*) = delta_{ij} - <j_|i> + delta_{ij}<j_|i>
		 */

		// indices for basis determinants
		span span_doa = span(2, n_vir);
		span span_idv = span(n_vir+1, n_vir+n_occ-1);

		// indices for orbitals
		span span_occ = span(1, n_occ-1);
		span span_vir = span(n_occ+1, n_occ+n_vir-1);

		
		ovl(0,0) = 1.0;
		ovl(0,1) = ovl_orb(0, n_occ);
		ovl(0, span_doa) = ovl_orb(0, span_vir);
		ovl(0, span_idv) = ovl_orb(span_occ, n_occ).t();

		ovl(1,0) = ovl_orb(n_occ, 0);
		ovl(1,1) = 1;
		ovl(1, span_doa) = ovl_orb(n_occ, span_vir);
		ovl(1, span_idv) = -ovl_orb(span_occ, 0).t();

		ovl(span_doa, 0) = ovl_orb(span_vir, 0);
		ovl(span_doa, 1) = ovl_orb(span_vir, n_occ);
		ovl(span_doa, span_doa) = ovl_orb(span_vir, span_vir);
		ovl(span_doa, span_idv) = zeros(n_vir-1, n_occ-1);

		ovl(span_idv, 0) = ovl_orb(n_occ, span_occ).t();
		ovl(span_idv, 1) = -ovl_orb(0, span_occ).t();
		ovl(span_idv, span_doa) = zeros(n_occ-1, n_vir-1);
		ovl(span_idv, span_idv) = eye(n_occ-1, n_occ-1) - ovl_orb(span_occ, span_occ).t() + eye(n_occ-1, n_occ-1) % ovl_orb(span_occ, span_occ).t();
	}

	return ovl;
}


vec det12(mat const& A, uvec const& i, mat const& vecs, char const& rc) {
	if ( rc == 'c' ) {
		arma::mat C = solve(A, vecs);
		return det(A) * vectorise(C.rows(i));
	}
	arma::mat C = solve(A.t(), vecs.t()).t();
	return det(A) * vectorise(C.cols(i));
}

vec det3(mat const& A, uword const& i, mat const& u, uvec const& j, mat const& v, char const& rc) {
	mat cu, cv;
	if ( rc == 'c' ) {
		cu = solve(A, u);
		cv = solve(A, v);
	} else {
		cu = solve(A.t(), u.t()).t();
		cv = solve(A.t(), v.t()).t();
	}
	return det(A) * ( cu(i) * cv(j) - cu(j) * cv(i) ); // always column!
}


