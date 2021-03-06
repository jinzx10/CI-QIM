#include "FSSH.h"
#include "math_helper.h"
#include "arma_helper.h"

using namespace arma;

FSSH::FSSH( 
		TwoPara2*					model_,
		double			const&		mass_,
		double			const&		dtc_,
		uword			const& 		ntc_,
		double			const&		kT_,
		double			const&		gamma_
):
	model(model_), mass(mass_), dtc(dtc_), ntc(ntc_),
	kT(kT_), gamma(gamma_),
	x(0), v(0), F_pes(0), sz( model->sz_elec ), span_exc( span(1,sz-1) ),
	state(0), rho(zeros<cx_mat>(sz, sz)),
	E_adi(zeros(sz)), rho_eq(zeros(sz)),
	counter(0), has_hop(false),
	x_t(zeros(ntc)), v_t(zeros(ntc)), E_t(zeros(ntc)),
	state_t(zeros<uvec>(ntc)), num_frustrated_hops(0)
{}

void FSSH::initialize(bool const& state0, double const& x0, double const& v0, cx_mat const& rho0) {
	clear();
	state = state0;
	x = x0;
	v = v0;
	rho = rho0;
	E_adi = model->E_adi(x);
	rho_eq = exp(-(E_adi-E_adi(0))/kT) / accu( exp(-(E_adi-E_adi(0))/kT) );
	collect();
}

void FSSH::evolve_nucl() {
	// Velocity-Verlet (with external phononic friction)
	double F_fric = -gamma * v;
	double F_rand = sqrt( 2.0 * gamma * kT / dtc ) * randn();
	F_pes = model->force(state, x);
	double a = ( F_pes + F_fric + F_rand ) / mass;
	x += v * dtc + 0.5 * a * dtc * dtc;
	F_pes = model->force(state, x);
	double a_new = ( F_pes + F_fric + F_rand ) / mass;
	v += 0.5 * (a + a_new) * dtc;

	// time derivative coupling matrix
	T = v * model->dc(x);
	
	// instantaneous adiabatic energies and equilibrium population
	E_adi = model->E_adi(x);
	rho_eq = exp(-(E_adi-E_adi(0))/kT) / accu( exp(-(E_adi-E_adi(0))/kT) );
	Gamma_rlx = vec{model->Gamma(x)};
}

void FSSH::calc_dtq() {
	double dtq1 = 0.02 / abs(T).max();
	double dtq2 = 0.02 / abs(E_adi - mean(E_adi)).max();
	dtq = min(dtc, dtq1, dtq2);
	rcq = dtc / dtq;
	dtq = (rcq > 1) ? dtc / rcq : dtc;
}

double FSSH::energy() {
	double E_kin = 0.5 * mass * v * v;
	double E_elec = E_adi(state);
	return E_kin + E_elec;
}

cx_mat FSSH::L_rho(cx_mat const& rho_) {
	cx_mat tmp = zeros<cx_mat>(sz, sz);

	vec L_diag = zeros(sz);
	vec rho_diag = real(rho_.diag());

	L_diag(span_exc) = Gamma_rlx % ( rho_diag(span_exc) - rho_eq(span_exc) );
	L_diag(0) = -accu( L_diag(span_exc) );

	tmp.diag() = conv_to<cx_vec>::from(L_diag);
	tmp(0, span_exc) = 0.5 * Gamma_rlx.t() % rho_(0, span_exc);
	tmp(span_exc, 0) = 0.5 * Gamma_rlx % rho_(span_exc, 0);
	return tmp;
}

cx_mat FSSH::drho_dt(cx_mat const& rho_) {
	std::complex<double> I{0.0, 1.0};
	return -I * rho_ % bcast_minus(E_adi, E_adi.t())
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
		q.tail(sz-1) = Gamma_rlx % ( rho_diag(span_exc) - rho_eq(span_exc) );
	}

	// hopping probability to each state
	vec g_hop = g + q;
	vec P_hop = dtq * g_hop % (g_hop > 0) / rho(state, state).real();

	// determine the destination state of hopping
	vec P_cumu = cumsum(P_hop);
	uword target = 0;
	arma_rng::set_seed_random();
	double r = randu();
	double r_base = 0;
	for (target = 0; target != sz; ++target) {
		if ( r < P_cumu(target) )
			break;
		r_base = P_cumu(target);
	}

	if ( target == sz ) // no hopping happens
		return;

	// various velocity-reversal schemes
	int opt_velorev = 0; // default, standard velocity reversal
	bool hop_from_dc = true; // used if PARTIAL_VELOCITY_REVERSAL is set

#ifdef PARTIAL_VELOCITY_REVERSAL
	opt_velorev = 1;
	if (g(target) < 0) {
		hop_from_dc = false;
	} else {
		double dr = r - r_base;
		if ( dr/P_hop(target) > g(target)/(g(target)+q(target)) )
			hop_from_dc = false;
	}
#endif

#ifdef NO_VELOCITY_REVERSAL
	opt_velorev = 2;
#endif

	double dE = E_adi(target) - E_adi(state);
	if ( dE <= 0.5 * mass * v * v) { // successful hops
		v = v_sign * std::sqrt(v*v - 2.0 * dE / mass);
		state = target;
		has_hop = 1;
	} else { // frustrated hops
		num_frustrated_hops += 1;
		if ( opt_velorev == 0 || (opt_velorev == 1 && hop_from_dc) ) {
			double F_tmp = model->force(target, x);
			if ( F_pes*F_tmp < 0 && F_tmp*v < 0  )
				v = -v;
		}
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
	num_frustrated_hops = 0;
}

void FSSH::propagate() {
	for (counter = 1; counter != ntc; ++counter) {
		evolve_nucl();
		calc_dtq();
		has_hop = 0;
		for (uword i = 0; i != rcq; ++i) {
			evolve_elec();
			if (!has_hop)
				hop();
		}
		collect();
	}
}

