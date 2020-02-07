#ifndef __BORN_OPPENHEIMER_DYNAMICS_H__
#define __BORN_OPPENHEIMER_DYNAMICS_H__

#include <armadillo>
#include <TwoPara_interp.h>

struct BO
{
	BO( 
			TwoPara_interp*				model_,
			double			const&		mass_,
			double			const&		dtc_,
			arma::uword		const& 		ntc_,
			double			const& 		kT_,
			double 			const& 		gamma_
	);

	void						initialize(double const& x0_, double const& v0_);
	void						propagate();

	void						evolve_nucl(); // Velocity Verlet
	void 						collect();
	void						clear();

	double						energy();

	TwoPara_interp*	const		model;
	double			const		mass;
	double 			const		dtc;
	arma::uword		const 		ntc; // number of classical time steps
	double			const		kT;
	double 			const		gamma; // external phononic friction

	double						x;
	double 						v;

	arma::uword					counter;

	// data storage for one trajectory
	arma::vec					x_t;
	arma::vec					v_t;
	arma::vec					E_t;
};

#endif


