#ifndef __FEWEST_SWITCHES_SURFACE_HOPPING_H__
#define __FEWEST_SWITCHES_SURFACE_HOPPING_H__

#include <TwoPara.h>
#include <armadillo>

struct FSSH
{
	FSSH( TwoPara* model_,
			double  const& mass_,
			double const& dt_,
			unsigned int const& nt_,
			double const& kT_,
			double const& gamma_	);

	void initialize(bool const& state0_, double const& x0_, double const& v0_, arma::cx_mat const& denmat0_);
	void propagate();

	// three tasks in one progagation step
	void			rk4_onestep();
	void 			hop();
	void 			collect();

	// first-order differential equation
	arma::vec		dvar_dt(arma::vec const& var_);

	TwoPara*		model;
	double mass;
	double dt;
	unsigned int nt;
	double kT;
	double gamma;

	arma::uword state;
	arma::vec var;
	unsigned int counter;

	// data storage for one trajectory
	arma::vec		x_t;
	arma::vec		v_t;
	arma::vec		rho00_t;
	arma::vec		Re_rho01_t;
	arma::vec		Im_rho01_t;
	arma::uvec		state_t;
};

#endif
