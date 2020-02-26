#include "SIAM.h"
#include "math_helper.h"

using namespace arma;

SIAM::SIAM(
		vec const& bath_,
		vec const& cpl_,
		double const& U_,
		uword const& n_occ_
): 
	bath(bath_), cpl(cpl_), U(U_), n_bath(bath.n_elem), n_occ(n_occ_), n_vir(n_bath+1-n_occ)
{
	h = diagmat(join_cols(vec{0}, bath));
	h(0, span(1, n_bath)) = cpl.t();
	h(span(1, n_bath), 0) = cpl;
	n_imp_mf = 0;
}

void SIAM::set_Ed(double const& Ed) {
	h(0,0) = Ed;
}

mat SIAM::F(double const& n) {
	mat Fock = h;
	Fock(0,0) += U * n;
	return Fock;
}

double SIAM::n2n(double const& n) {
	mat Fock = F(n);
	mat eigvec;
	vec eigval;
	eig_sym(eigval, eigvec, Fock);
	return accu(square(eigvec(0, span(0, n_occ-1))));
}

void SIAM::solve_mf() {
	auto dn = [this] (double const& n) { return n2n(n) - n; };
	newtonroot(dn, n_imp_mf);
	eig_sym(val_mf, vec_mf, F(n_imp_mf));
	E_mf = accu(val_mf.head(n_occ)) - U * n_imp_mf * n_imp_mf;
}




