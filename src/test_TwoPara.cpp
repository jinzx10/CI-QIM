#include "../include/TwoPara.h"
#include <functional>

using namespace arma;

int main() {
	double x0_mpt = 2;
	double x0_fil = 2.3;
	double omega= 0.002;
	double mass = 14000;
	double dE_fil = 0.0000;
	
	auto E_mpt = [&](double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_mpt) * (x - x0_mpt);};
	auto E_fil = [&](double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_fil) * (x - x0_fil) + dE_fil;};

	double W = 0.05;
	double bath_min = -W;
	double bath_max = W;
	uword n_bath = 800;
	vec bath = linspace<vec>(bath_min, bath_max, n_bath);
	double dos = 1.0 / (bath(1) - bath(0));

	double Gamma = 0.001;
	auto cpl = [&](double const& x) -> vec {
		return ones<vec>(n_bath) * sqrt(Gamma/2/datum::pi/dos);
	};

	uword n_occ = n_bath / 2;

	TwoPara model(E_mpt, E_fil, bath, cpl, n_occ);

	return 0;
}
