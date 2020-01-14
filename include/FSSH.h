#ifndef __FEWEST_SWITCHES_SURFACE_HOPPING_H__
#define __FEWEST_SWITCHES_SURFACE_HOPPING_H__

#include <TwoPara.h>
#include <armadillo>

struct FSSH
{
	FSSH( TwoPara* model_,
			double const& mass_,
			double const& dtc_,
			unsigned int const& rcq_, // ratio between dtc and dtq
			unsigned int const& ntc_,
			double const& kT_,
			double const& gamma_ // external phononic friction
	);

	void initialize(bool const& state0_, double const& x0_, double const& v0_, arma::cx_mat const& rho0_);
	void propagate();

	void solve();
	void evolve_nucl(); // Velocity Verlet
	void calc_T();
	void evolve_elec(); // Runge-Kutta
	void hop();
	void collect();


	TwoPara*		model;
	double mass;
	double dtc;
	double dtq;
	unsigned int rcq;
	unsigned int ntc;
	double kT;
	double gamma;

	arma::uword state;
	double x;
	double v;
	arma::cx_mat rho;
	unsigned int counter;

	bool has_hop;

	// data storage for one trajectory
	arma::vec		x_t;
	arma::vec		v_t;
	arma::uvec		state_t;
};

#endif
