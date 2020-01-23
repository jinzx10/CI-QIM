#ifndef __FEWEST_SWITCHES_SURFACE_HOPPING_H__
#define __FEWEST_SWITCHES_SURFACE_HOPPING_H__

#include <TwoPara.h>
#include <armadillo>

struct FSSH
{
	FSSH(	TwoPara*					model_,
			double			const&		mass_,
			double			const&		dtc_,
			arma::uword		const& 		rcq_, // ratio between dtc and dtq
			arma::uword		const& 		ntc_,
			double			const& 		kT_,
			double 			const& 		gamma_ // external phononic friction
	);

	void				initialize(bool const& state0_, double const& x0_, double const& v0_, arma::cx_mat const& rho0_);
	void				propagate();

	void				evolve_nucl(); // Velocity Verlet
	void 				calc_T();
	void 				evolve_elec(); // Runge-Kutta
	void 				hop();
	void 				collect();
	arma::cx_mat		drho_dt(arma::cx_mat const& rho_);

	TwoPara*			model;
	double				mass;
	double 				dtc;
	double 				dtq;
	arma::uword			rcq;
	arma::uword 		ntc;
	double				kT;
	double 				gamma;

	arma::uword			state;
	double				x;
	double 				v;
	arma::cx_mat		rho; // density matrix
	arma::mat			T; // time-derivative matrix, <p|(d/dt)q>

	arma::uword			counter;
	bool				has_hop;

	// data storage for one trajectory
	arma::vec			x_t;
	arma::vec			v_t;
	arma::uvec			state_t;
};

#endif
