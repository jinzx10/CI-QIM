#ifndef __MODEL_INTERPOLATION_H__
#define __MODEL_INTERPOLATION_H__

#include <armadillo>

struct ModelInterp
{
	ModelInterp(	arma::vec	const&		xgrid_,
					arma::mat	const&		pes_,
					arma::mat	const&		force_,
					arma::mat	const&		drvcpl_,
					arma::mat	const&		Gamma_		);

	double				E(double const& x, arma::uword const& state);
	arma::vec			E(double const& x);
	double				F(double const& x, arma::uword const& state);
	arma::vec			F(double const& x);
	arma::mat			dc(double const& x);
	arma::vec			Gamma(double const& x);

	arma::uword			sz_elec;

	private:
	arma::vec			_xgrid;
	arma::mat			_pes;
	arma::mat			_force;
	arma::mat			_drvcpl;
	arma::mat			_Gamma;
};

#endif
