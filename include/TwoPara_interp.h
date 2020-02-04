#ifndef __TWO_PARABOLA_INTERPOLATION_H__
#define __TWO_PARABOLA_INTERPOLATION_H__

#include <armadillo>

struct TwoPara_interp
{
	TwoPara_interp(		arma::vec	const&		xgrid_,
						arma::vec 	const&		E0_,
						arma::vec 	const&		E1_,
						arma::vec 	const&		F0_,
						arma::vec 	const&		F1_,
						arma::vec 	const&		dc01_,
						arma::vec 	const&		Gamma_		);

	double				E_adi(arma::uword const& state, double const& x);
	double				force(arma::uword const& state, double const& x);
	double				dc(arma::uword const& state_i, arma::uword const& state_j, double const& x);

	double				E0(double const& x);
	double				E1(double const& x);
	double				F0(double const& x);
	double				F1(double const& x);
	double				dc01(double const& x);
	double				Gamma(double const& x);

	private:
	arma::vec			xgrid_;
	arma::vec			E0_;
	arma::vec			E1_;
	arma::vec			F0_;
	arma::vec			F1_;
	arma::vec			dc01_;
	arma::vec			Gamma_;
};

#endif
