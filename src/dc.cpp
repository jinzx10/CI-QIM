#include <dc.h>
#include <join.h>

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

uvec range(uword const& i, uword const& j) {
	return regspace<uvec>(i, j); // end-inclusive
}

mat ovl(vec const& vec_do_, mat const& vec_occ_, vec const& vec_dv_, mat const& vec_vir_, vec const& vec_do, mat const& vec_occ, vec const& vec_dv, mat const& vec_vir) {
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


	vo.save("/home/zuxin/job/CI-QIM/data/test_dc/vo.txt", raw_ascii);
	vv.save("/home/zuxin/job/CI-QIM/data/test_dc/vv.txt", raw_ascii);
	ovl_orb.save("/home/zuxin/job/CI-QIM/data/test_dc/orb_overlap.txt", raw_ascii);


	// sizes and indices of the orbitals:
	//           do     occ           dv           vir 
	// size	     1     n_occ-1        1          n_vir-1
	// index     0    1 : n_occ-1    n_occ    n_occ+1 : n_occ+n_vir-1
	//
	// sizes and indices of the basis Slater determinants:
	//          Psi_{0}  Psi_{do}^{dv}   Psi_{do}^{a}     Psi_{i}^{dv}
	// size       1            1           n_vir-1          n_occ-1
	// index      0            1          2 : n_vir   n_vir+1 : n_vir+n_occ-1
	
	// indices for basis determinants
	span span_doa = span(1, n_vir); // dv is included in vir
	span span_idv = span(n_vir+1, n_vir+n_occ-1);

	// indices for orbitals
	span span_o = span(0, n_occ-1);
	uvec idx_o = range(0, n_occ-1);
	uvec idx_occ = range(1, n_occ-1);
	span span_v = span(n_occ, n_occ+n_vir-1);
	uvec idx_v = range(n_occ, n_occ+n_vir-1);

	// first row
	mat ref = ovl_orb(span_o, span_o);
	ovl(0, 0) = det(ref);
	std::cout << "00 fin" << std::endl;
	ovl(0, span_doa) = det12(ref, uvec{0}, ovl_orb(span_o, span_v)).t();
	std::cout << "doa fin" << std::endl;
	ovl(0, span_idv) = det12(ref, idx_occ, ovl_orb(span_o, n_occ)).t();
	std::cout << "idv fin" << std::endl;

	// first column
	ovl(span_doa, 0) = det12(ref, uvec{0}, ovl_orb(span_v, span_o), 'r');
	ovl(span_idv, 0) = det12(ref, idx_occ, ovl_orb(n_occ, span_o), 'r');

	std::cout << "ab block begin" << std::endl;
	// ab block
	for (uword a = 0; a != n_vir; ++a) {
		uvec idx_ref = idx(n_occ+a, range(1, n_occ-1));
		ref = ovl_orb(idx_ref, idx_ref);
		ovl(1+a, span_doa) = det12(ref, uvec{0}, ovl_orb(idx_ref, idx_v)).t();
	}

	std::cout << "ij block begin" << std::endl;
	// ij block
	for (uword i = 1; i != n_occ; ++i) {
		uvec idx_ref = idx(range(0, i-1), n_occ, range(i+1, n_occ-1));
		ref = ovl_orb(idx_ref, idx_ref);
		ovl(n_vir+i, span_idv) = -det12(ref, idx_occ, ovl_orb(idx_ref, uvec{i})).t();
		ovl(n_vir+i, n_vir+i) = det(ref);
	}

	std::cout << "aj block begin" << std::endl;
	// aj block
	for (uword a = 0; a != n_vir; ++a) {
		uvec idx_ref = idx(n_occ+a, range(1, n_occ-1));
		ref = ovl_orb(idx_ref, idx_ref);
		ovl(a+1, span_idv) = det3(ref, 0, ovl_orb(idx_ref, uvec{0}), idx_occ, ovl_orb(idx_ref, uvec{n_occ})).t();

	}

	std::cout << "ib block begin" << std::endl;
	// ib block
	for (uword b = 0; b != n_vir; ++b) {
		uvec idx_ref = idx(n_occ+b, range(1, n_occ-1));
		ref = ovl_orb(idx_ref, idx_ref);
		ovl(span_idv, b+1) = det3(ref, 0, ovl_orb(uvec{0}, idx_ref), idx_occ, ovl_orb(uvec{n_occ}, idx_ref), 'r');
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
