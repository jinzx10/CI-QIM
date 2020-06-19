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
		double			const&		gamma_,
		int				const&		velo_rev_,
		int 			const& 		has_rlx_,
		uword 			const& 		sz_elec_
):
	model(model_), mass(mass_), dtc(dtc_), ntc(ntc_),
	kT(kT_), gamma(gamma_), velo_rev(velo_rev_), has_rlx(has_rlx_),
	x(0), v(0), F_pes(0), 
	sz_elec( (sz_elec_ && sz_elec_ <= model->sz_elec) ?  
			sz_elec_ : model->sz_elec ), 
	span_exc( span(1,sz_elec-1) ),
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
	E_adi = model->E(x).head(sz_elec);
	rho_eq = boltzmann(E_adi, kT);
	collect();
	arma::arma_rng::set_seed_random();
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
	E_adi = model->E(x).head(sz_elec);
	rho_eq = boltzmann(E_adi, kT);
	Gamma_rlx = model->Gamma(x).head(sz_elec);
}

void FSSH_rlx::calc_dtq() {
	double dtq1 = 0.02 / abs(T).max();
	double dtq2 = 0.02 / abs(E_adi - mean(E_adi)).max();
	dtq = min(dtc, dtq1, dtq2);
	rcq = dtc / dtq;
	dtq = (rcq > 1) ? dtc / rcq : dtc;
}

double FSSH_rlx::energy() {
	return 0.5*mass*v*v + E_adi(state);
}

/*
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
*/

cx_mat FSSH_rlx::Lindblad(cx_mat const& rho_) {
	cx_mat tmp = zeros<cx_mat>(sz_elec, sz_elec);
	vec f = 1.0 / (exp( (E_adi-E_adi(0))/kT ) + 1.0);
	vec rho_diag = real(rho_.diag());
	vec sqrtf1f = sqrt( f % (1.0-f) );
	vec G1f = Gamma_rlx % (1.0-f);
	double Gf = accu(Gamma_rlx % f);

	tmp(0,0) += accu( Gamma_rlx % (1.0-f) % rho_diag );
	tmp.diag() += Gamma_rlx % f * rho_(0,0);
	tmp.row(0) += (Gamma_rlx % sqrtf1f % rho_.col(0)).as_row();
	tmp.col(0) += Gamma_rlx % sqrtf1f % (rho_.row(0).as_col());

	tmp.row(0) -= 0.5 * Gf * rho_.row(0);
	tmp.col(0) -= 0.5 * Gf * rho_.col(0);
	tmp -= 0.5 * bcast_plus(G1f, G1f.as_row()) % rho_;

	/*
	sp_mat L(sz_elec, sz_elec); // jump operator
	double f = 0.0;
	for (size_t i = 1; i != sz_elec; ++i) {
		L.zeros();
		f = 1.0 / ( exp((E_adi(i)-E_adi(0))/kT) + 1.0 );
		L(0,i) = sqrt(1.0-f);
		L(i,0) = sqrt(f);
		tmp += Gamma_rlx(i) * (L*rho_*L.t() - 0.5*(L.t()*L*rho_+rho_*(L.t()*L)));
	}
	*/
	return tmp;
}

cx_mat FSSH_rlx::drho_dt(cx_mat const& rho_) {
	std::complex<double> I{0.0, 1.0};
	if (has_rlx)
		return -I * rho_ % bcast_minus(E_adi, E_adi.t())
		   	- (T * rho_ - rho_ * T) + Lindblad(rho_);
	return -I * rho_ % bcast_minus(E_adi, E_adi.t()) - (T * rho_ - rho_ * T);
	//return -I * rho_ % bcast_minus(E_adi, E_adi.t())
	//	- (T * rho_ - rho_ * T) - L_rho(rho_);
}

void FSSH_rlx::evolve_elec() {
	cx_mat k1 = dtq * drho_dt(rho);
	cx_mat k2 = dtq * drho_dt(rho + 0.5*k1);
	cx_mat k3 = dtq * drho_dt(rho + 0.5*k2);
	cx_mat k4 = dtq * drho_dt(rho + k3);
	rho += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
}

void FSSH_rlx::hop() {
	int v_sign = (v >= 0) ? 1 : -1;

	// (d/dt)rho_mm = -\sum_l ( g_lm + q_lm )
	// g and q are anti-symmetric
	// g_lm = 2*Re(T_ml*rho_lm)
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
	//arma_rng::set_seed_random();
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

	// hopping happens, check if frustrated or not
	double dE = E_adi(fs) - E_adi(state);
	if ( fs < state || dE <= 0.5 * mass * v * v ) { // successful hops
		v = v_sign * std::sqrt(v*v - 2.0 * dE / mass);
		state = fs;
		has_hop = true;
	} else { // frustrated hops
		num_frustrated_hops += 1;

		// for frustrated hops, check if the velocity should be reversed
		// various velocity-reversal schemes
		double F_fs;
		switch (velo_rev) {
			case 0: // standard
				F_fs = model->F(x, fs);
				v = ( F_pes*F_fs < 0 && F_fs*v < 0 ) ? -v : v;
				break;
			case 1: // partial, check whether hop from dc or rlx
				if ( g(fs) > 0 && (dr/P_hop(fs) < g(fs)/(g(fs)+q(fs))) ) {
					F_fs = model->F(x, fs);
					v = ( F_pes*F_fs < 0 && F_fs*v < 0 ) ? -v : v;
				}
				break;
			default: // no velocity reversal
				;
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


