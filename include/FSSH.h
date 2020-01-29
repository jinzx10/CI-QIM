#ifndef __FEWEST_SWITCHES_SURFACE_HOPPING_H__
#define __FEWEST_SWITCHES_SURFACE_HOPPING_H__

#include <TwoPara.h>
#include <armadillo>

struct FSSH
{
	FSSH(	TwoPara*					model_,
			double			const&		mass_,
			double			const&		dtc_,
			arma::uword		const& 		ntc_,
			double			const& 		kT_,
			double 			const& 		gamma_
	);

	void				initialize(bool const& state0_, double const& x0_, double const& v0_, arma::cx_mat const& rho0_);
	void				propagate();

	void				evolve_nucl(); // Velocity Verlet
	void 				evolve_elec(); // Runge-Kutta
	void				calc_dtq();
	void 				hop();
	void 				collect();
	void				clear();

	double				energy();
	arma::cx_mat		L_rho(arma::cx_mat const& rho_);
	arma::cx_mat		drho_dt(arma::cx_mat const& rho_);

	TwoPara*			model;
	double				mass;
	double 				dtc;
	double 				dtq; // calculated after each classical time step
	arma::uword			rcq; // dtc/dtq
	arma::uword 		ntc; // number of classical time steps
	double				kT;
	double 				gamma; // external phononic friction

	arma::uword			state;
	double				x;
	double 				v;
	double				F_pes;
	arma::cx_mat		rho; // density matrix
	arma::mat			T; // time-derivative matrix, <p|(d/dt)q>
	arma::uword			sz; // size of the electronic basis
	arma::span			span_cis;
	arma::vec			E_adi;
	arma::vec			rho_eq; // instantaneous equilibrium population

	arma::uword			counter;
	bool				has_hop;

	// data storage for one trajectory
	arma::vec			x_t;
	arma::vec			v_t;
	arma::uvec			state_t;
	arma::vec			E_t;
};

#endif
