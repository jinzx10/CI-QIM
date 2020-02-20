#ifndef __TWO_PARABOLA_INTERPOLATION_H__
#define __TWO_PARABOLA_INTERPOLATION_H__

#include <armadillo>

struct TwoPara2
{
	TwoPara2(	arma::vec	const&		xgrid_,
				arma::vec 	const&		E0_,
				arma::vec 	const&		E1_,
				arma::vec 	const&		F0_,
				arma::vec 	const&		F1_,
				arma::vec 	const&		dc01_,
				arma::vec 	const&		Gamma_		);

	double				E_adi(arma::uword const& state, double const& x);
	arma::vec			E_adi(double const& x);
	double				force(arma::uword const& state, double const& x);
	arma::mat			dc(double const& x);

	double				E0(double const& x);
	double				E1(double const& x);
	double				F0(double const& x);
	double				F1(double const& x);
	double				dc01(double const& x);
	double				Gamma(double const& x);

	arma::uword			sz_elec = 2;

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
