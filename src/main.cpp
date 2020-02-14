#include <mpi.h>
#include <chrono>
#include <cstdlib>
#include "FSSH_interp.h"
#include "TwoPara_interp.h"
#include "arma_mpi_helper.h"
#include "arma_helper.h"

using namespace arma;
using iclock = std::chrono::high_resolution_clock;

int main() {

	int id, nprocs;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	iclock::time_point start;
	std::chrono::duration<double> dur;

	std::string param = "00128";

	std::string readdir = "/home/zuxin/job/CI-QIM/data/test_TwoPara/Gamma/";
	std::string savedir = "/home/zuxin/job/CI-QIM/data/main/Gamma/";

	readdir += param + "/";

#ifdef NO_VELOCITY_REVERSAL
	savedir += param + "_norev/";
#else
	savedir += param + "/";
#endif


	std::string command = "mkdir -p " + savedir;


	////////////////////////////////////////////////////////////
	//					Two-Parabola model
	////////////////////////////////////////////////////////////

	vec E0, E1, F0, F1, dc01, Gamma, xgrid;
	uword sz;
	if (id == 0) {
		start = iclock::now();

		arma_load(readdir, 
				xgrid, "xgrid.txt",
				E0, "E0.txt", 
				E1, "E1.txt",
				F0, "F0.txt", 
				F1, "F1.txt",
				dc01, "dc01.txt", 
				Gamma, "Gamma.txt"
		);

		sz = xgrid.n_elem;
	}

	bcast(&sz);

	if (id != 0) {
		set_size(sz, xgrid, E0, E1, F0, F1, dc01, Gamma);
	}

	bcast(xgrid, E0, E1, F0, F1, dc01, Gamma);

	TwoPara_interp model(xgrid, E0, E1, F0, F1, arma::abs(dc01), Gamma);


	////////////////////////////////////////////////////////////
	//			Fewest-Switches Surface Hopping
	////////////////////////////////////////////////////////////

	double omega = 0.0002;
	double mass = 2000;
	double x0_mpt = 0;

	double t_max = 1e5;
	double dtc = 10;
	int n_trajs = 960;
	uword ntc = t_max / dtc;
	vec time_grid;
	if (id == 0) {
		time_grid = linspace(0, t_max, ntc);
		dtc = time_grid(1) - time_grid(0);
	}
	bcast(&dtc);

	double fric_gamma = 2.0 * mass * omega;
	double kT = 9.5e-4;

	int n_trajs_local = n_trajs / nprocs;
	arma::uword sz_rho = 2;
	FSSH_interp fssh_interp(&model, mass, dtc, ntc, kT, fric_gamma);

	// local data
	mat x_local, v_local, E_local;
	umat state_local;
	uvec n_fhop_local;

	set_size(ntc, n_trajs_local, x_local, v_local, state_local, E_local);
	set_size(n_trajs_local, n_fhop_local);

	// global data
	mat x_t, v_t, E_t;
	umat state_t;
	uvec n_fhop;

	if (id == 0) {
		set_size(ntc, n_trajs, x_t, v_t, state_t, E_t);
		set_size(n_trajs, n_fhop);
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
		n_fhop_local(i) = fssh_interp.num_frustrated_hops;
	}

	gather( state_local, state_t, x_local, x_t, v_local, v_t,
			E_local, E_t, n_fhop_local, n_fhop );

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

		arma_save<raw_binary>( savedir, 
				state_t, "state.dat", 
				x_t, "x.dat",
				v_t, "v.dat",
				E_t, "E.dat",
				n_fhop, "fhop.dat",
				time_grid, "t.dat"
		);

		dur = iclock::now() - start;
		std::cout << dur.count() << std::endl;
	}

	::MPI_Finalize();

	return 0;
}
