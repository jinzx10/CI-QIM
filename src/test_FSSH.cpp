#include <FSSH.h>
#include <TwoPara.h>
#include <chrono>
#include <cstdlib>

using namespace arma;
using iclock = std::chrono::high_resolution_clock;

int main() {

	////////////////////////////////////////////////////////////
	//					Two-Parabola model
	////////////////////////////////////////////////////////////
	double x0_mpt = 2;
	double x0_fil = 2.3;
	double omega = 0.002;
	double mass = 14000;
	double dE_fil = 0.0000;
	
	auto E_mpt = [&] (double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_mpt) * (x - x0_mpt);};
	auto E_fil = [&] (double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_fil) * (x - x0_fil) + dE_fil;};

	double W = 0.05;
	double bath_min = -W;
	double bath_max = W;
	uword n_bath = 400;
	vec bath = linspace<vec>(bath_min, bath_max, n_bath);
	double dos = 1.0 / (bath(1) - bath(0));

	uword n_occ = n_bath / 2;
	uword n_vir = n_bath + 1 - n_occ;
	uword sz_sub = n_occ + n_vir;

	double hybrid = 0.001;
	auto cpl = [&] (double const& x) -> vec {
		return ones<vec>(n_bath) * sqrt(hybrid/2/datum::pi/dos);
	};

	TwoPara model(E_mpt, E_fil, bath, cpl, n_occ);

	////////////////////////////////////////////////////////////
	//			Fewest-Switches Surface Hopping
	////////////////////////////////////////////////////////////
	double dtc = 1;
	uword rcq = 1.0 / omega / 50;
	uword ntc = 1000;
	double kT = 0.001;
	double fric_gamma = 2.0 * mass * omega;
	FSSH fssh(&model, mass, dtc, rcq, ntc, kT, fric_gamma);

	uword state0 = 0; 
	double x0 = 2.1;
	double v0 = 3.0 * sqrt(kT/mass);
	cx_mat rho0 = zeros<cx_mat>(sz_sub, sz_sub);
	rho0(0,0) = 1.0;
	fssh.initialize(state0, x0, v0, rho0);

	fssh.propagate();

	////////////////////////////////////////////////////////////
	//					save data
	////////////////////////////////////////////////////////////
	std::string datadir = "/home/zuxin/job/CI-QIM/data/test_FSSH/";
	std::string command = "mkdir -p " + datadir;
	bool status = std::system(command.c_str());
	if (!status)
		datadir = "";

	fssh.state_t.save(datadir + "state.txt", raw_ascii);
	fssh.x_t.save(datadir + "x.txt", raw_ascii);
	fssh.v_t.save(datadir + "v.txt", raw_ascii);


	return 0;
}
