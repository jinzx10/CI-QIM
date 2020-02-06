#include <mpi.h>
#include <FSSH_interp.h>
#include <TwoPara_interp.h>
#include <arma_mpi_helper.h>
#include <chrono>
#include <cstdlib>

using namespace arma;
using iclock = std::chrono::high_resolution_clock;

int main() {

	int id, nprocs;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	iclock::time_point start;
	std::chrono::duration<double> dur;

	std::string param = "0000025_1000";
	std::string savedir = "/home/zuxin/job/CI-QIM/data/test_FSSH_interp/Gamma/";
	savedir += param + "/";
	std::string command = "mkdir -p " + savedir;
	////////////////////////////////////////////////////////////
	//					Two-Parabola model
	////////////////////////////////////////////////////////////
	std::string readdir = "/home/zuxin/job/CI-QIM/data/test_TwoPara/Gamma/";
	readdir += param + "/";

	vec E0, E1, F0, F1, dc01, Gamma, xgrid;
	uword sz;
	if (id == 0) {
		start = iclock::now();

		xgrid.load(readdir + "xgrid.txt");
		E0.load(readdir + "E0.txt");
		E1.load(readdir + "E1.txt");
		F0.load(readdir + "F0.txt");
		F1.load(readdir + "F1.txt");
		dc01.load(readdir + "dc01.txt");
		Gamma.load(readdir + "Gamma.txt");

		sz = xgrid.n_elem;
	}

	bcast(&sz);

	if (id != 0) {
		xgrid.zeros(sz);
		E0.zeros(sz);
		E1.zeros(sz);
		F0.zeros(sz);
		F1.zeros(sz);
		dc01.zeros(sz);
		Gamma.zeros(sz);
	}

	bcast(xgrid, E0, E1, F0, F1, dc01, Gamma);

	TwoPara_interp model(xgrid, E0, E1, F0, F1, arma::abs(dc01), Gamma);

	////////////////////////////////////////////////////////////
	//			Fewest-Switches Surface Hopping
	////////////////////////////////////////////////////////////
	double omega = 0.0002;
	double mass = 2000;
	double x0_mpt = 0;

#ifdef DEBUG_MODE
	int n_trajs = 96;
	uword ntc = 20;
	double fric_gamma = 0;
#else
	int n_trajs = 960;
	uword ntc = 100000;
	double fric_gamma = 2.0 * mass * omega;
#endif

	double dtc = 20;
	double kT = 9.5e-4;
	int n_trajs_local = n_trajs / nprocs;

	arma::uword sz_rho = 2;
	FSSH_interp fssh_interp(&model, mass, dtc, ntc, kT, fric_gamma);

	// local data
	mat x_local = arma::zeros(ntc, n_trajs_local);
	mat v_local = arma::zeros(ntc, n_trajs_local);
	umat state_local = arma::zeros<umat>(ntc, n_trajs_local);
	mat E_local = arma::zeros(ntc, n_trajs_local);

	// global data
	mat x_t;
	mat v_t;
	umat state_t;
	mat E_t;

	if (id == 0) {
		x_t.zeros(ntc, n_trajs);
		v_t.zeros(ntc, n_trajs);
		state_t.zeros(ntc, n_trajs);
		E_t.zeros(ntc, n_trajs);
		
		start = iclock::now();
	}

	// Wigner quasi-probability distribution of harmonic oscillator
	// ground state
	//double sigma_x = std::sqrt(0.5/mass/omega);
	//double sigma_v = std::sqrt(omega/mass/2.0);
	
	// thermally-averaged
	double sigma_x = std::sqrt( 0.5 / mass / omega / std::tanh(omega/2.0/kT) );
	double sigma_p = std::sqrt( mass * omega / 2.0 / std::tanh(omega/2.0/kT) );
	arma::arma_rng::set_seed_random();

	for (int i = 0; i != n_trajs_local; ++i) {
		double x0 = x0_mpt + arma::randn()*sigma_x;
		double v0 = arma::randn() * sigma_p / mass;

		vec val = model.E_adi(x0);
		vec rho_eq = exp(-(val-val(0))/kT) / accu( exp(-(val-val(0))/kT) );
		cx_mat rho0 = zeros<cx_mat>(sz_rho, sz_rho);
		rho0(0,0) = rho_eq(0);
		rho0(1,1) = rho_eq(1);
		uword state0 = (arma::randu() < rho_eq(0)) ? 0 : 1;

		fssh_interp.initialize(state0, x0, v0, rho0);
		fssh_interp.propagate();

		x_local.col(i) = fssh_interp.x_t;
		v_local.col(i) = fssh_interp.v_t;
		state_local.col(i) = fssh_interp.state_t;
		E_local.col(i) = fssh_interp.E_t;
	}

	gather(state_local, state_t, x_local, x_t, v_local, v_t, E_local, E_t);

	////////////////////////////////////////////////////////////
	//					save data
	////////////////////////////////////////////////////////////
	if (id == 0) {
		bool status = std::system(command.c_str());
		if (status) {
			savedir = "./";
			command = "mkdir -p " + savedir;
			status = std::system(command.c_str());
		}

		state_t.save(savedir + "state.txt", raw_ascii);
		x_t.save(savedir + "x.txt", raw_ascii);
		v_t.save(savedir + "v.txt", raw_ascii);
		E_t.save(savedir + "E.txt", raw_ascii);

		dur = iclock::now() - start;
		std::cout << dur.count() << std::endl;
	}

	::MPI_Finalize();

	return 0;
}
