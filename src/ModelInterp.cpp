#include "ModelInterp.h"
#include "math_helper.h"

using namespace arma;

ModelInterp::ModelInterp(
		vec		const&		xgrid_,
		mat		const&		pes_,
		mat		const&		force_,
		mat		const&		drvcpl_,
		mat		const&		Gamma_ 
): 
	sz_elec(pes_.n_cols),
	_xgrid(xgrid_), _pes(pes_), _force(force_), 
	_drvcpl(drvcpl_), _Gamma(Gamma_) 
{}


double ModelInterp::E(double const& x, uword const& state) {
	return lininterp(x, _xgrid, vec{_pes.col(state)});
}

vec	ModelInterp::E(double const& x) {
	return lininterp(x, _xgrid, _pes).as_col();
}

double ModelInterp::F(double const& x, uword const& state) {
	return lininterp(x, _xgrid, vec{_force.col(state)});
}

vec ModelInterp::F(double const& x) {
	return lininterp(x, _xgrid, _force).as_col();
}

mat ModelInterp::dc(double const& x) {
	return reshape(lininterp(x, _xgrid, _drvcpl), sz_elec, sz_elec);
}

vec ModelInterp::Gamma(double const& x) {
	return lininterp(x, _xgrid, _Gamma).as_col();
}
