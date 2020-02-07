#include <TwoPara.h>
#include <chrono>
#include <mpi.h>
#include <cstdlib>
#include <arma_mpi_helper.h>
#include <arma_helper.h>
#include <auxmath.h>

using namespace arma;
using iclock = std::chrono::high_resolution_clock;

int main() {

	int id, nprocs;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	iclock::time_point start;
	std::chrono::duration<double> dur;
	std::string datadir = "/home/zuxin/job/CI-QIM/data/test_TwoPara/Gamma/00128/";
	//std::string datadir = "/home/zuxin/job/CI-QIM/data/test_TwoPara/";
	std::string cmd;
	int status;

	////////////////////////////////////////////////////////////
	//					Two-Parabola model
	////////////////////////////////////////////////////////////
	double x0_mpt = 0;
	double x0_fil = 20.6097;
	double omega = 0.0002;
	double mass = 2000;
	double dE_fil = -0.0038;
	
	auto E_mpt = [&] (double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_mpt) * (x - x0_mpt);};
	auto E_fil = [&] (double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_fil) * (x - x0_fil) + dE_fil;};

	double W = 0.1;
	double bath_min = -W;
	double bath_max = W;
	uword n_bath = 1000;
	vec bath = linspace<vec>(bath_min, bath_max, n_bath);
	double dos = 1.0 / (bath(1) - bath(0));

	uword n_occ = n_bath / 2;
	uword n_vir = n_bath + 1 - n_occ;
	uword sz_sub = n_occ + n_vir - 1;

	double hybrid = 0.0128;
	vec cpl = ones<vec>(n_bath) * sqrt(hybrid/2/datum::pi/dos);

	//uword nx = 32*3*10;
	//vec xgrid = linspace(-20, 40, nx);
	double dw = 2.8;
	auto x_density = [dw] (double x) { 
		return 9.01 + 32.2 * exp(-(x-8.0)*(x-8.0)/2.0/dw/dw);
	};
	vec xgrid = grid(-20, 40, x_density);
	uword nx = xgrid.n_elem;

	int nx_local = nx / nprocs;

	// global variables (only proc 0 will initialize them)
	vec E0, E1, F0, F1, dc01, Gamma, n_imp;

	if (id == 0) {
		set_size( nx, E0, E1, F0, F1, dc01, Gamma, n_imp );
		start = iclock::now();
	}


	// local variables and their initialization
	vec E0_local, E1_local, F0_local, F1_local,
		dc01_local, Gamma_local, n_imp_local;

	set_size( nx_local, E0_local, E1_local, F0_local, F1_local,
			dc01_local, Gamma_local, n_imp_local );


	// Two parabola model
	TwoPara model(E_mpt, E_fil, bath, cpl, n_occ);

	for (int i = 0; i != nx_local; ++i) {
		double x = xgrid(id*nx_local+i);
		model.set_and_calc_cis_sub(x);
		model.calc_val_cis_bath();
		model.calc_Gamma(1);

		E0_local(i) = model.ev_H + E_mpt(x);
		E1_local(i) = model.val_cis_sub(0) + E_mpt(x);
		F0_local(i) = model.force(0);
		F1_local(i) = model.force(1);
		dc01_local(i) = model.dc(2)(0,1);
		Gamma_local(i) = model.Gamma(0);
		n_imp_local(i) = model.ev_n;

		std::cout << id*nx_local+i+1 << "/" << nx << " finished" << std::endl;
	}

	gather( E0_local, E0, E1_local, E1, F0_local, F0, F1_local, F1,
			dc01_local, dc01, Gamma_local, Gamma, n_imp_local, n_imp );

	if (id == 0) {
		cmd = "mkdir -p " + datadir;
		status = std::system(cmd.c_str());

		arma_save<raw_ascii>( datadir,
				xgrid, "xgrid.txt", 
				E0, "E0.txt", 
				E1, "E1.txt",
				F0, "F0.txt",
				F1, "F1.txt",
				dc01, "dc01.txt", 
				Gamma, "Gamma.txt",
				n_imp, "n_imp.txt"
		);

		dur = iclock::now() - start;
		std::cout << "time elapsed = " << dur.count() << std::endl;
	}

	MPI_Finalize();
	
	return 0;
}
