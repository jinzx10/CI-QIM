#include <FSSH.h>
#include <TwoPara.h>
#include <chrono>
#include <cstdlib>
#include <mpi.h>

using namespace arma;
using iclock = std::chrono::high_resolution_clock;

int main() {

	int id, nprocs;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	iclock::time_point start;
	std::chrono::duration<double> dur;

	std::string datadir = "/home/zuxin/job/CI-QIM/data/test_FSSH/";
	std::string command = "mkdir -p " + datadir;

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
	int n_trajs = 192;
	int n_trajs_local = n_trajs / nprocs;

	double dtc = 20;
	double dtq_est = 0.2;
	uword rcq = dtc / dtq_est;
	uword ntc = 200;
	double kT = 0.005;
	double fric_gamma = 2.0 * mass * omega;
	FSSH fssh(&model, mass, dtc, rcq, ntc, kT, fric_gamma);

	// local data
	mat x_local = arma::zeros(ntc, n_trajs_local);
	mat v_local = arma::zeros(ntc, n_trajs_local);
	umat state_local = arma::zeros<umat>(ntc, n_trajs_local);

	// global data
	mat x_t;
	mat v_t;
	umat state_t;

	if (id == 0) {
		x_t.zeros(ntc, n_trajs);
		v_t.zeros(ntc, n_trajs);
		state_t.zeros(ntc, n_trajs);
		
		start = iclock::now();
	}

	// Wigner quasi-probability of the harmonic ground state:
	// exp(-m*omega*x^2/hbar) * exp(-p^2/m/omega/hbar)
	double sigma_x = std::sqrt(0.5/mass/omega);
	double sigma_v = std::sqrt(omega/mass/2.0);
	arma::arma_rng::set_seed_random();

	for (int i = 0; i != n_trajs_local; ++i) {
		uword state0 = 0; 
		double x0 = x0_mpt + arma::randn()*sigma_x;
		double v0 = arma::randn()*sigma_v;
		cx_mat rho0 = zeros<cx_mat>(sz_sub, sz_sub);
		rho0(0,0) = 1.0;
		fssh.initialize(state0, x0, v0, rho0);
		fssh.propagate();

		x_local.col(i) = fssh.x_t;
		v_local.col(i) = fssh.v_t;
		state_local.col(i) = fssh.state_t;
	}

	::MPI_Gather(state_local.memptr(), state_local.n_elem, MPI_UNSIGNED_LONG_LONG, state_t.memptr(), state_local.n_elem, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
	::MPI_Gather(x_local.memptr(), x_local.n_elem, MPI_DOUBLE, x_t.memptr(), x_local.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	::MPI_Gather(v_local.memptr(), v_local.n_elem, MPI_DOUBLE, v_t.memptr(), v_local.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	////////////////////////////////////////////////////////////
	//					save data
	////////////////////////////////////////////////////////////
	if (id == 0) {
		bool status = std::system(command.c_str());
		if (status)
			datadir = "";

		state_t.save(datadir + "state.txt", raw_ascii);
		x_t.save(datadir + "x.txt", raw_ascii);
		v_t.save(datadir + "v.txt", raw_ascii);

		dur = iclock::now() - start;
		std::cout << dur.count() << std::endl;
	}

	::MPI_Finalize();

	return 0;
}
