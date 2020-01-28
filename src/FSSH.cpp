#include <FSSH.h>
#include <dc.h>
#include <join.h>
#include <complex>
#include <fermi.h>
#include <chrono>
#include <bcast_op.h>

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
	x_t(zeros(ntc)), v_t(zeros(ntc)), state_t(zeros<uvec>(ntc)), E_t(zeros(ntc))
{
	dtq = dtc / rcq;
	sz = model->sz_rel;
	span_cis = span(1, sz-1);
}

void FSSH::initialize(bool const& state0, double const& x0, double const& v0, arma::cx_mat const& rho0) {
	state = state0;
	x = x0;
	v = v0;
	rho = rho0;
	clear();
	model->set_and_calc(x);
	collect();
}

void FSSH::evolve_nucl() {
	// store the necessary data for the time derivative coupling calculation
	vec do_ = model->vec_do;
	vec dv_ = model->vec_dv;
	mat vec_o_ = model->vec_o;
	mat vec_v_ = model->vec_v;
	mat coef_ = join<mat>( { { vec{1}		 , zeros(1, sz-1)	   },
							 { zeros(sz-1, 1), model->vec_cis_sub  } } );

	// Velocity-Verlet (with external phononic friction)
	double F_fric = -gamma * v;
	double F_rand = sqrt( 2.0 * gamma * kT / dtc ) * randn();
	F_pes = model->force(state);
	double a = ( F_pes + F_fric + F_rand ) / mass;
	x += v * dtc + 0.5 * a * dtc * dtc;
	model->set_and_calc(x);
	F_pes = model->force(state);
	double a_new = ( F_pes + F_fric + F_rand ) / mass;
	v += 0.5 * (a + a_new) * dtc;

	// calculate the time derivative coupling
	mat coef = join<mat>( { { vec{1}		, zeros(1, sz-1)	  },
							{ zeros(sz-1, 1), model->vec_cis_sub  } } );
	adj_phase(coef_, coef);
	mat overlap = coef_.t() * ovl(do_, vec_o_, dv_, vec_v_, model->vec_do, model->vec_o, model->vec_dv, model->vec_v) * coef;

	// Lowdin-orthoginalization
	overlap *= sqrtmat_sympd( overlap.t() * overlap );

	// time derivative coupling matrix
	T = real( logmat(overlap) ) / dtc;
	
	// instantaneous adiabatic energies and equilibrium population
	rho_eq = exp(-model->E_rel() / kT) / accu( exp(-model->E_rel() / kT) );
}

double FSSH::energy() {
	double E_kin = 0.5 * mass * v * v;
	double E_pot = model->E_mpt(x);
	double E_elec = model->E_rel(state);
	return E_kin + E_pot + E_elec;
}

cx_mat FSSH::L_rho(cx_mat const& rho_) {
	cx_mat tmp = zeros<cx_mat>(sz, sz);

	vec L_diag = zeros(sz);
	vec rho_diag = real(rho_.diag());
	L_diag(span_cis) = model->Gamma % ( rho_diag(span_cis) - rho_eq(span_cis) );
	L_diag(0) = -accu( L_diag(span_cis) );

	tmp.diag() = conv_to<cx_vec>::from(L_diag);
	tmp(0, span_cis) = 0.5 * model->Gamma.t() % rho_(0, span_cis);
	tmp(span_cis, 0) = 0.5 * model->Gamma % rho_(span_cis, 0);
	return tmp;
}

cx_mat FSSH::drho_dt(cx_mat const& rho_) {
	std::complex<double> I{0.0, 1.0};
	return -I * rho_ % bcast_op(model->E_rel(), model->E_rel().t(), std::minus<>()) - (T * rho_ - rho_ * T) - L_rho(rho_);
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
		q(span_cis) = model->Gamma % ( rho_diag(span_cis) - rho_eq(span_cis) );
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

	double dE = model->E_rel(target) - model->E_rel(state);
	if ( dE <= 0.5 * mass * v * v) { // successful hops
		v = v_sign * std::sqrt(v*v - 2.0 * dE / mass);
		state = target;
		has_hop = 1;
	} else { // frustrated hops
		double F_tmp = model->force(target);
		if ( F_pes*F_tmp < 0 && F_tmp*v < 0  ) // velocity reveral
			v = -v;
	}
}

void FSSH::collect() {
	state_t(counter) = state;
	x_t(counter) = x;
	v_t(counter) = v;
	E_t(counter) = energy();
}

void FSSH::clear() {
	counter = 0;
	x_t.zeros();
	v_t.zeros();
	state_t.zeros();
	E_t.zeros();
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
		std::cout << counter << "/" << ntc << " finished" << std::endl;
	}
}

