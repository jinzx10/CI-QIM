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
				double			const&		gamma_	):
	model(model_), mass(mass_), dtc(dtc_), rcq(rcq_), ntc(ntc_),
	kT(kT_), gamma(gamma_),
	state(0), x(0), v(0), rho(cx_mat{}), counter(0), has_hop(false),
	x_t(zeros(ntc)), v_t(zeros(ntc)), state_t(zeros<uvec>(ntc))
{
	dtq = dtc / rcq;
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
	cx_mat L_rho = zeros<cx_mat>(sz, sz);
	vec tmp = zeros(sz);
	vec pop = real(rho_.diag());
	pop.shed_row(0);
	tmp.tail(sz-1) = model->Gamma % (pop-rho_eq);
	tmp(0) = -accu(tmp.tail(sz-1));
	L_rho.diag() = conv_to<cx_vec>::from(tmp);
	L_rho(0, span(1, sz-1)) = model->Gamma.t() % rho_(0, span(1, sz-1));
	L_rho(span(1, sz-1), 0) = model->Gamma % rho_(span(1, sz-1), 0);
	
	cx_mat drho = -I * ( rho_.each_col() % conv_to<cx_vec>::from(E) - rho_.each_row() % conv_to<cx_vec>::from(E) ) - (T * rho_ - rho_ * T) - L_rho;

	return drho;
}

void FSSH::evolve_elec() {
	cx_vec rho0 = vectorise(rho);
	uword sz = rho.n_cols;

	cx_vec k1 = dtq * drho_dt(rho0);
	cx_vec k2 = dtq * drho_dt(rho0 + 0.5*k1);
	cx_vec k3 = dtq * drho_dt(rho0 + 0.5*k2);
	cx_vec k4 = dtq * drho_dt(rho0 + k3);

	rho0 += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
	rho = arma::reshape(rho0, sz, sz);
}

void FSSH::hop() {
	arma::arma_rng::set_seed_random();
	double r = randu();
	int v_sign = (v >= 0) * 1 + (v < 0) * -1;

	// find hopping amplitudes to various states
	// by assumption, for an excited state, it either stays or hops to the ground state
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

