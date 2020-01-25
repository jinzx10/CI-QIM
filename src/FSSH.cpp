#include <FSSH.h>
#include <dc.h>
#include <join.h>
#include <complex>
#include <fermi.h>

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
	sz_elec = model->n_occ + model->n_vir;
	L_rho.zeros(sz_elec, sz_elec);
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
	double a = model->force_(x,state) / mass;
	x += v * dtc + 0.5 * a * dtc * dtc;
	double a_new = model->force_(x,state) / mass;
	v += 0.5 * (a + a_new) * dtc;
}

void FSSH::calc_T() {
	vec do_ = model->vec_do;
	vec dv_ = model->vec_dv;
	mat vec_occ_ = model->vec_occ;
	mat vec_vir_ = model->vec_vir;

	mat coef_ = model->vec_cis_sub;
	uword sz = coef_.n_cols;
	coef_ = join<mat>( { {vec{1}, zeros(1, sz)}, {zeros(sz, 1), coef_} } );

	model->set_and_calc(x);

	mat coef = join<mat>( { {vec{1}, zeros(1, sz)},
			{zeros(sz, 1), model->vec_cis_sub} } );

	mat overlap = coef_.t() * ovl(do_, vec_occ_, dv_, vec_vir_, model->vec_do, model->vec_occ, model->vec_dv, model->vec_vir) * coef;

	// Lowdin-orthoginalization
	overlap *= sqrtmat_sympd( overlap.t() * overlap );

	// time derivative matrix
	T = real( logmat(overlap) ) / dtc;
}

cx_mat FSSH::drho_dt(cx_mat const& rho_) {
	uword sz = rho_.n_cols;
	vec E = join_cols( vec{model->ev_H}, model->val_cis_sub );
	std::complex<double> I{0.0, 1.0};

	vec rho_eq = exp(-E/kT) / accu(exp(-E/kT));
	
	vec L_diag = zeros(sz);
	vec rho_diag = real(rho_.diag());
	span idx_cis = span(1, sz-1);
	L_diag(idx_cis) = model->Gamma % ( rho_diag(idx_cis) - rho_eq(idx_cis) );
	L_diag(0) = -accu( L_diag(idx_cis) );

	L_rho.diag() = conv_to<cx_vec>::from(L_diag);
	L_rho(0, idx_cis) = 0.5 * model->Gamma.t() % rho_(0, idx_cis);
	L_rho(idx_cis, 0) = 0.5 * model->Gamma % rho_(idx_cis, 0);
	
	return -I * ( rho_.each_col() % conv_to<cx_vec>::from(E) - rho_.each_row() % conv_to<cx_rowvec>::from(E.t()) ) - (T * rho_ - rho_ * T) - L_rho;
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
	vec q = zeros(sz_elec); // extra damping
	if (state) {
		q(0) = L_rho(state, state).real();
	} else {
		q = -real( L_rho.diag() );
		q(0) = 0;
	}

	// hopping probability to each state
	vec g_hop = g + q;
	vec P_hop = dtq * g_hop % (g_hop > 0) / rho(state, state).real();

	// determine the destination state of hopping
	vec P_cumu = cumsum(P_hop);
	uword target = 0;
	arma_rng::set_seed_random();
	double r = randu();
	for (target = 0; target != sz_elec; ++target) {
		if ( r < P_cumu(target) )
			break;
	}

	if ( target == sz_elec ) // no hopping happens
		return;








	if (state) { // excited -> ground
		double g_hop = drho_dt(rho)(state, state).real() / rho(state, state).real() * dtq;
		if ( r < g_hop ) {
			state = 0;
			has_hop = 1;
			double dE = model->val_cis_sub(state-1) - model->ev_H;
			v = v_sign * sqrt(v*v + 2.0 * dE / mass);
		}
	} else { // ground -> excited
		vec drho_tmp = real( drho_dt(rho).diag() );
		drho_tmp.shed_row(0);
		vec rho_tmp = real( rho.diag() );
		rho_tmp.shed_row(0);
		vec g_hops = drho_tmp / rho_tmp * dtq;
		vec g_cumsum = cumsum(g_hops);
		for (uword i = 0; i != g_cumsum.n_elem; ++i) {
			if ( r > g_cumsum(i) && r < g_cumsum(i+1)/*problem!*/ ) {
				double dE = model->val_cis_sub(i+1) - model->ev_H;
				if ( 0.5 * mass * v * v > dE ) {
					v = v_sign * std::sqrt( v*v - 2.0 * dE / mass );
					state = i+1;
					has_hop = 1;
				} else { // frustrated hop: velocity reversal

				}
				break;
			}
		}

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
		calc_T();
		has_hop = 0;
		for (arma::uword i = 0; i != rcq; ++i) {
			evolve_elec();
			if (!has_hop)
				hop();
		}
		collect();
	}
}

