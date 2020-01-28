#include <FSSH.h>
#include <dc.h>
#include <join.h>
#include <complex>
#include <fermi.h>
#include <chrono>

using namespace arma;

FSSH::FSSH(		TwoPara*					model_,
				double			const&		mass_,
				double			const&		dtc_,
				arma::uword		const& 		rcq_,
				arma::uword		const& 		ntc_,
				double			const&		kT_,
				double			const&		gamma_		):
	model(model_), mass(mass_), dtc(dtc_), rcq(rcq_), ntc(ntc_),
	kT(kT_), gamma(gamma_),
	state(0), x(0), v(0), rho(cx_mat{}), counter(0), has_hop(false),
	x_t(zeros(ntc)), v_t(zeros(ntc)), state_t(zeros<uvec>(ntc))
{
	dtq = dtc / rcq;
	sz = model->n_occ + model->n_vir;
	idx_cis = span(1, sz-1);
}

void FSSH::initialize(bool const& state0, double const& x0, double const& v0, arma::cx_mat const& rho0) {
	state = state0;
	x = x0;
	v = v0;
	rho = rho0;
	counter = 0;

	x_t.zeros();
	v_t.zeros();
	state_t.zeros();

	model->set_and_calc(x);

	collect();
}

void FSSH::evolve_nucl() {
	// store the necessary data for time derivative coupling calculation
	vec do_ = model->vec_do;
	vec dv_ = model->vec_dv;
	mat vec_occ_ = model->vec_occ;
	mat vec_vir_ = model->vec_vir;
	mat coef_ = join<mat>( { { vec{1}		 , zeros(1, sz-1)	   },
							 { zeros(sz-1, 1), model->vec_cis_sub  } } );

	// Velocity-Verlet (with external phononic friction)
	double F_fric = -gamma * v;
	double F_rand = sqrt( 2.0 * gamma * kT / dtc ) * randn();
	F_pes = model->force_(state);
	double a = ( F_pes + F_fric + F_rand ) / mass;
	x += v * dtc + 0.5 * a * dtc * dtc;
	model->set_and_calc(x);
	F_pes = model->force_(state);
	double a_new = ( F_pes + F_fric + F_rand ) / mass;
	v += 0.5 * (a + a_new) * dtc;

	// calculate the time derivative coupling
	mat coef = join<mat>( { { vec{1}		, zeros(1, sz-1)	  },
							{ zeros(sz-1, 1), model->vec_cis_sub  } } );
	mat overlap = coef_.t() * ovl(do_, vec_occ_, dv_, vec_vir_, model->vec_do, model->vec_occ, model->vec_dv, model->vec_vir) * coef;

	// Lowdin-orthoginalization
	overlap *= sqrtmat_sympd( overlap.t() * overlap );

	// time derivative coupling matrix
	T = real( logmat(overlap) ) / dtc;
	
	// instantaneous adiabatic energies and equilibrium population
	E = join_cols( vec{model->ev_H}, model->val_cis_sub );
	rho_eq = exp(-E/kT) / accu( exp(-E/kT) );
}

cx_mat FSSH::L_rho(cx_mat const& rho_) {
	cx_mat tmp = zeros<cx_mat>(sz, sz);

	vec L_diag = zeros(sz);
	vec rho_diag = real(rho_.diag());
	L_diag(idx_cis) = model->Gamma % ( rho_diag(idx_cis) - rho_eq(idx_cis) );
	L_diag(0) = -accu( L_diag(idx_cis) );

	tmp.diag() = conv_to<cx_vec>::from(L_diag);
	tmp(0, idx_cis) = 0.5 * model->Gamma.t() % rho_(0, idx_cis);
	tmp(idx_cis, 0) = 0.5 * model->Gamma % rho_(idx_cis, 0);
	return tmp;
}

cx_mat FSSH::drho_dt(cx_mat const& rho_) {
	std::complex<double> I{0.0, 1.0};
	return -I * ( rho_.each_col() % conv_to<cx_vec>::from(E) - 
				  rho_.each_row() % conv_to<cx_rowvec>::from(E.t()) ) 
		- (T * rho_ - rho_ * T) - L_rho(rho_);
}

void FSSH::evolve_elec() {
	cx_mat k1 = dtq * drho_dt(rho);
	cx_mat k2 = dtq * drho_dt(rho + 0.5*k1);
	cx_mat k3 = dtq * drho_dt(rho + 0.5*k2);
	cx_mat k4 = dtq * drho_dt(rho + k3);
	rho += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
}

void FSSH::hop() {
	int v_sign = (v >= 0) * 1 + (v < 0) * -1;

	// individual components in (d/dt)rho_mm
	vec g = 2.0 * real( T.row(state).t() % rho.col(state) ); // normal 
	vec q = zeros(sz); // extra damping
	if (state) {
		q(0) = model->Gamma(state-1) * ( rho(state, state).real() - rho_eq(state) );
	} else {
		vec rho_diag = real(rho.diag());
		q(idx_cis) = model->Gamma % ( rho_diag(idx_cis) - rho_eq(idx_cis) );
	}

	// hopping probability to each state
	vec g_hop = g + q;
	vec P_hop = dtq * g_hop % (g_hop > 0) / rho(state, state).real();

	// determine the destination state of hopping
	vec P_cumu = cumsum(P_hop);
	uword target = 0;
	arma_rng::set_seed_random();
	double r = randu();
	for (target = 0; target != sz; ++target) {
		if ( r < P_cumu(target) )
			break;
	}

	if ( target == sz ) // no hopping happens
		return;

	double dE = E(target) - E(state);
	if ( dE <= 0.5 * mass * v * v) { // successful hops
		v = v_sign * std::sqrt(v*v - 2.0 * dE / mass);
		state = target;
		has_hop = 1;
	} else { // frustrated hops
		double F_tmp = model->force_(x, target);
		if ( F_pes*F_tmp < 0 && F_tmp*v < 0  ) // velocity reveral
			v = -v;
	}
}

void FSSH::collect() {
	state_t(counter) = state;
	x_t(counter) = x;
	v_t(counter) = v;
}

void FSSH::propagate() {
	for (counter = 1; counter != ntc; ++counter) {
		evolve_nucl();
		has_hop = 0;
		for (arma::uword i = 0; i != rcq; ++i) {
			evolve_elec();
			if (!has_hop)
				hop();
		}
		collect();
	}
}

