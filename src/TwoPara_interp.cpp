#include <TwoPara_interp.h>
#include <interp.h>

using namespace arma;

TwoPara_interp::TwoPara_interp(
		vec		const&		xgrid_,
		vec 	const&		E0_,
		vec 	const&		E1_,
		vec 	const&		F0_,
		vec 	const&		F1_,
		vec 	const&		dc01_,
		vec 	const&		Gamma_		):
	xgrid_(xgrid_), E0_(E0_), E1_(E1_), F0_(F0_), F1_(F1_),
	dc01_(dc01_), Gamma_(Gamma_)
{}

double TwoPara_interp::E0(double const& x) {
	return lininterp_linspace(x, xgrid_, E0_);
}

double TwoPara_interp::E1(double const& x) {
	return lininterp_linspace(x, xgrid_, E1_);
}

double TwoPara_interp::F0(double const& x) {
	return lininterp_linspace(x, xgrid_, F0_);
}

double TwoPara_interp::F1(double const& x) {
	return lininterp_linspace(x, xgrid_, F1_);
}

double TwoPara_interp::Gamma(double const& x) {
	return lininterp_linspace(x, xgrid_, Gamma_);
}

double TwoPara_interp::dc01(double const& x) {
	return lininterp_linspace(x, xgrid_, dc01_);
}

double TwoPara_interp::E_adi(uword const& state, double const& x) {
	return (state) ? E1(x) : E0(x);
}

vec TwoPara_interp::E_adi(double const& x) {
	return vec{E0(x), E1(x)};
}

double TwoPara_interp::force(uword const& state, double const& x) {
	return (state) ? F1(x) : F0(x);
}

mat TwoPara_interp::dc(double const& x) {
	double d01 = dc01(x);
	return mat{ {0.0, d01}, {-d01, 0.0} };
}


