#include <dc.h>

using namespace arma;

void adj_phase(mat const& vecs_old, mat& vecs_new) {
	mat ovl = vecs_old.t() * vecs_new;
	for (uword j = 0; j != ovl.n_cols; ++j) {
		uword i = abs(ovl.col(j)).index_max();
		if ( ovl(i,j) < 0 ) {
			vecs_new.col(j) *= -1;
		}
	}
}

cx_mat dc(vec const& vec_do_, mat const& vec_occ_, vec const& vec_dv_, mat const& vec_vir_, vec const& vec_do, mat const& vec_occ, vec const& vec_dv, mat const& vec_vir) {
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
	
	// dv is included in vir
	span idx_doa = span(1, n_vir);
	span idx_idv = span(n_vir+1, n_vir+n_occ-1);
	span idx_o = span(0, n_occ-1);
	span idx_v = span(n_occ, n_occ+n_vir-1);

	// end-inclusive range
	auto range = [](uword const& i, uword const& j) -> uvec {
		return regspace<uvec>(i, j);
	};

	// first row
	mat ref = ovl_orb(idx_o, idx_o);
	ovl(0,0) = det(ref);
	ovl(0, idx_doa) = det12(ref, 0, ovl_orb(idx_o, idx_v)).t();
	ovl(0, idx_idv) = det12(ref, range(1, n_occ-1), ovl_orb(idx_o, n_occ)).t();

	// first column
	ovl(idx_doa, 0) = det12(ref, 0, ovl_orb(idx_v, idx_o), 'r');
	ovl(idx_idv, 0) = det12(ref, range(1, n_occ-1), ovl_orb(n_occ, idx_o), 'r');

	// ab block

	// ij block

	// aj block

	// ib block


	return logmat(ovl);
}


vec det12(mat const& A, uvec const& idx, mat const& vecs, char const& rc) {
	if ( rc == 'c' ) {
		arma::mat C = solve(A, vecs);
		return det(A) * vectorise(C.rows(idx));
	}
	arma::mat C = solve(A.t(), vecs.t()).t();
	return det(A) * vectorise(C.cols(idx));
}

vec det3(mat const& A, uword const& i, mat const& u, uvec const& idx, mat const& v, char const& rc) {
	mat cu, cv;
	if ( rc == 'c' ) {
		cu = solve(A, u);
		cv = solve(A, v);
	} else {
		cu = solve(A.t(), u.t()).t();
		cv = solve(A.t(), v.t()).t();
	}
	return det(A) * ( cu(i) * cv(idx) - cu(idx) * cv(i) ); // always column!
}

