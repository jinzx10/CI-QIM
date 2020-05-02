#include "FSSH_rlx.h"
#include "math_helper.h"
#include "arma_helper.h"

using namespace arma;

FSSH_rlx::FSSH_rlx( 
		ModelInterp*				model_,
		double			const&		mass_,
		double			const&		dtc_,
		uword			const& 		ntc_,
		double			const&		kT_,
		double			const&		gamma_
):
	model(model_), mass(mass_), dtc(dtc_), ntc(ntc_),
	kT(kT_), gamma(gamma_), 
	x(0), v(0), F_pes(0), 
	sz_elec( model->sz_elec ), span_exc( span(1,sz_elec-1) ),
	state(0), rho(zeros<cx_mat>(sz_elec, sz_elec)),
	E_adi(zeros(sz_elec)), rho_eq(zeros(sz_elec)),
	counter(0), has_hop(false),
	x_t(zeros(ntc)), v_t(zeros(ntc)), E_t(zeros(ntc)),
	state_t(zeros<uvec>(ntc)), num_frustrated_hops(0)
{}

void FSSH_rlx::initialize(bool const& state0, double const& x0, double const& v0, cx_mat const& rho0) {
	clear();
	x = x0;
	v = v0;
	state = state0;
	rho = rho0;
	E_adi = model->E(x);
	rho_eq = exp(-(E_adi-E_adi(0))/kT) / accu( exp(-(E_adi-E_adi(0))/kT) );
	collect();
}

void FSSH_rlx::evolve_nucl() {
	// Velocity-Verlet (with external phononic friction)
	double F_fric = -gamma * v;
	double F_rand = sqrt( 2.0 * gamma * kT / dtc ) * randn();
	F_pes = model->F(x, state);
	double a = ( F_pes + F_fric + F_rand ) / mass;
	x += v * dtc + 0.5 * a * dtc * dtc;
	F_pes = model->F(x, state);
	double a_new = ( F_pes + F_fric + F_rand ) / mass;
	v += 0.5 * (a + a_new) * dtc;

	// time derivative coupling matrix
	T = v * model->dc(x);
	
	// instantaneous adiabatic energies and equilibrium population
	E_adi = model->E(x);
	rho_eq = exp(-(E_adi-E_adi(0))/kT) / accu( exp(-(E_adi-E_adi(0))/kT) );
	Gamma_rlx = model->Gamma(x);
}

void FSSH_rlx::calc_dtq() {
	double dtq1 = 0.02 / abs(T).max();
	double dtq2 = 0.02 / abs(E_adi - mean(E_adi)).max();
	dtq = min(dtc, dtq1, dtq2);
	rcq = dtc / dtq;
	dtq = (rcq > 1) ? dtc / rcq : dtc;
}

double FSSH_rlx::energy() {
	double E_kin = 0.5 * mass * v * v;
	double E_elec = E_adi(state);
	return E_kin + E_elec;
}

cx_mat FSSH_rlx::L_rho(cx_mat const& rho_) {
	cx_mat tmp = zeros<cx_mat>(sz_elec, sz_elec);

	vec L_diag = zeros(sz_elec);
	vec rho_diag = real(rho_.diag());

	L_diag = Gamma_rlx % ( rho_diag - rho_eq );
	L_diag(0) = -accu( L_diag(span_exc) );

	tmp.diag() = conv_to<cx_vec>::from(L_diag);
	tmp.row(0) = 0.5 * Gamma_rlx.t() % rho_.row(0);
	tmp.col(0) = 0.5 * Gamma_rlx % rho_.col(0);
	return tmp;
}

cx_mat FSSH_rlx::drho_dt(cx_mat const& rho_) {
	std::complex<double> I{0.0, 1.0};
	return -I * rho_ % bcast_minus(E_adi, E_adi.t())
		- (T * rho_ - rho_ * T) - L_rho(rho_);
}

void FSSH_rlx::evolve_elec() {
	cx_mat k1 = dtq * drho_dt(rho);
	cx_mat k2 = dtq * drho_dt(rho + 0.5*k1);
	cx_mat k3 = dtq * drho_dt(rho + 0.5*k2);
	cx_mat k4 = dtq * drho_dt(rho + k3);
	rho += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
}

void FSSH_rlx::hop() {
	int v_sign = (v >= 0) * 1 + (v < 0) * -1;

	// (d/dt)rho_mm
	vec g = 2.0 * real( T.row(state).t() % rho.col(state) ); // normal 
	vec q = zeros(sz_elec); // extra damping
	vec rho_diag = real(rho.diag());
	if (state) {
		q(0) = Gamma_rlx(state) * ( rho_diag(state) - rho_eq(state) );
	} else {
		q = -Gamma_rlx % ( rho_diag - rho_eq );
	}

	// hopping probability to each state
	vec g_tot = g + q;
	vec P_hop = dtq * g_tot % (g_tot > 0) / rho_diag(state);

	// determine the final state of hopping
	vec P_cumu = cumsum(P_hop);
	uword fs = 0;
	arma_rng::set_seed_random();
	double r = randu();
	double dr = 0.0;
	for (fs = 0; fs != sz_elec; ++fs) {
		if ( r < P_cumu(fs) ) {
			dr = (fs) ? r - P_cumu(fs-1) : r;
			break;
		}
	}

	if ( fs == sz_elec ) // no hopping happens
		return;

	// various velocity-reversal schemes
	int opt_velorev = 0; // default, standard velocity reversal
	bool hop_from_dc = true; // used if PARTIAL_VELOCITY_REVERSAL is set

#ifdef PARTIAL_VELOCITY_REVERSAL
	opt_velorev = 1;
	if (g(fs) < 0) {
		hop_from_dc = false;
	} else {
		double dr = r - r_base;
		if ( dr/P_hop(fs) > g(fs)/(g(fs)+q(fs)) )
			hop_from_dc = false;
	}
#endif

#ifdef NO_VELOCITY_REVERSAL
	opt_velorev = 2;
#endif

	double dE = E_adi(fs) - E_adi(state);
	if ( dE <= 0.5 * mass * v * v) { // successful hops
		v = v_sign * std::sqrt(v*v - 2.0 * dE / mass);
		state = fs;
		has_hop = true;
	} else { // frustrated hops
		num_frustrated_hops += 1;
		if ( opt_velorev == 0 || (opt_velorev == 1 && hop_from_dc) ) {
			double F_tmp = model->F(x,fs);
			if ( F_pes*F_tmp < 0 && F_tmp*v < 0  )
				v = -v;
		}
	}
}

void FSSH_rlx::collect() {
	state_t(counter) = state;
	x_t(counter) = x;
	v_t(counter) = v;
	E_t(counter) = energy();
}

void FSSH_rlx::clear() {
	counter = 0;
	x_t.zeros();
	v_t.zeros();
	state_t.zeros();
	E_t.zeros();
	num_frustrated_hops = 0;
}

void FSSH_rlx::propagate() {
	for (counter = 1; counter != ntc; ++counter) {
		evolve_nucl();
		calc_dtq();
		has_hop = false;
		for (uword i = 0; i != rcq; ++i) {
			evolve_elec();
			if (!has_hop)
				hop();
		}
		collect();
	}
}


