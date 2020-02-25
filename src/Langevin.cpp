#include <Langevin.h>
#include <arma_helper.h>

using namespace arma;

Langevin::Langevin( 
		TwoPara2*					model_,
		double			const&		mass_,
		double			const&		dtc_,
		uword			const& 		ntc_,
		double			const&		kT_,
		double			const&		gamma_
):
	model(model_), mass(mass_), dtc(dtc_), ntc(ntc_),
	kT(kT_), gamma(gamma_), x(0), v(0), counter(0),
	x_t(zeros(ntc)), v_t(zeros(ntc)), E_t(zeros(ntc))
{}

void Langevin::initialize(double const& x0, double const& v0) {
	clear();
	x = x0;
	v = v0;
	collect();
}

void Langevin::evolve_nucl() {
	// Velocity-Verlet (with external phononic friction)
	double F_fric = -gamma * v;
	double F_rand = sqrt( 2.0 * gamma * kT / dtc ) * randn();
	double F_pes = model->F0(x);
	double a = ( F_pes + F_fric + F_rand ) / mass;
	x += v * dtc + 0.5 * a * dtc * dtc;
	F_pes = model->F0(x);
	double a_new = ( F_pes + F_fric + F_rand ) / mass;
	v += 0.5 * (a + a_new) * dtc;
}


double Langevin::energy() {
	double E_kin = 0.5 * mass * v * v;
	double E_elec = model->E0(x);
	return E_kin + E_elec;
}

void Langevin::collect() {
	x_t(counter) = x;
	v_t(counter) = v;
	E_t(counter) = energy();
}

void Langevin::clear() {
	counter = 0;
	x_t.zeros();
	v_t.zeros();
	E_t.zeros();
}

void Langevin::propagate() {
	for (counter = 1; counter != ntc; ++counter) {
		evolve_nucl();
		collect();
	}
}


