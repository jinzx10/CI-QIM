#ifndef __FEWEST_SWITCHES_SURFACE_HOPPING_WITH_RELAXATION_H__
#define __FEWEST_SWITCHES_SURFACE_HOPPING_WITH_RELAXATION_H__

#include <armadillo>
#include "ModelInterp.h"

struct FSSH_rlx
{
	FSSH_rlx( 
			ModelInterp*				model_,
			double			const&		mass_,
			double			const&		dtc_,
			arma::uword		const& 		ntc_,
			double			const& 		kT_,
			double 			const& 		gamma_,
			int				const&		velo_rev_,
			int 			const& 		velo_rescale_,
			int 			const&		has_rlx_ = 1, 
			arma::uword 	const& 		sz_elec_ = 0
	);

	void						initialize(bool const& state0_, double const& x0_, double const& v0_, arma::cx_mat const& rho0_);
	void						propagate();

	void						evolve_nucl(); // Velocity Verlet
	void 						evolve_elec(); // Runge-Kutta
	void						calc_dtq();
	void 						hop();
	void 						collect();
	void						clear();

	double						energy();
	//arma::cx_mat				L_rho(arma::cx_mat const& rho_);
	arma::cx_mat 				Lindblad(arma::cx_mat const& rho_);
	arma::cx_mat				drho_dt(arma::cx_mat const& rho_);

	ModelInterp*	const		model;
	double			const		mass;
	double 			const		dtc;
	double						dtq; // updated after each classical time step
	arma::uword					rcq; // int(dtc/dtq), at least 1
	arma::uword		const 		ntc; // number of classical time steps
	double			const		kT;
	double 			const		gamma; // external phononic friction
	int				const		velo_rev; // velocity reversal type
	int				const		velo_rescale; // velocity rescaling mode
	int 			const 		has_rlx; // has Lindblad term of not

	double						x;
	double 						v;
	double						F_pes;
	arma::uword		const		sz_elec; // size of the electronic basis
	arma::span		const		span_exc; // indices of excited states
	arma::uword					state;
	arma::cx_mat				rho; // density matrix
	arma::mat					T; // time-derivative matrix, <p|(d/dt)q>
	arma::vec					E_adi;
	arma::vec					rho_eq; // instantaneous equilibrium population
	arma::vec					Gamma_rlx; // relaxation Gamma

	arma::uword					counter; // counter for classical steps
	bool						has_hop; // for previous quantum steps

	// data storage for one trajectory
	arma::vec					x_t;
	arma::vec					v_t;
	arma::vec					E_t;
	arma::uvec					state_t;
	arma::vec 					n_t;
	arma::uword					num_frustrated_hops;
};

#endif

