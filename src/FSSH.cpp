#include <FSSH.h>
#include <dc.h>

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

void FSSH::calc_Tdc() {
	vec do_ = model->vec_do;
	vec dv_ = model->vec_dv;
	mat vec_occ_ = model->vec_occ;
	mat vec_vir_ = model->vec_vir;
	mat coef_ = model->vec_cis_sub;

	model->set_and_calc(x);

	mat overlap = ovl(do_, vec_occ_, dv_, vec_vir_, model->vec_do, model->vec_occ, model->vec_dv, model->vec_vir);

	// multiply coefficients
	// Lowdin-orthoginalization! 

	//Tdc = logmat(  ) / dtc;
}

cx_vec FSSH::drho_dt(cx_vec const& rho) {

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
	// find hopping amplitudes to various states
	// if hop, see if kinetic energy is enough
	// adjust velocity (if success), and set has_hop to 1
	// for frustrated hop, consider velocity reversal
}

void FSSH::collect() {
	state_t(counter) = state;
	x_t(counter) = x;
	v_t(counter) = v;
}

void FSSH::propagate() {
	for (counter = 1; counter != ntc; ++counter) {
		evolve_nucl();
		calc_Tdc();
		for (arma::uword i = 0; i != rcq; ++i) {
			evolve_elec();
			if (!has_hop)
				hop();
		}
		collect();
	}
}

