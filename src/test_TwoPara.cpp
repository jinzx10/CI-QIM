#include <TwoPara.h>
#include <chrono>
#include <mpi.h>
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
	std::string datadir = "/home/zuxin/job/CI-QIM/data/test_TwoPara/";
	std::string cmd;
	const char* system_cmd = nullptr;

	////////////////////////////////////////////////////////////
	//				Two-Parabola model
	////////////////////////////////////////////////////////////
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

	uword n_occ = n_bath / 2;
	uword n_vir = n_bath + 1 - n_occ;
	uword sz_sub = n_occ + n_vir - 1;

	double hybrid = 0.001;
	auto cpl = [&] (double const& x) -> vec {
		return ones<vec>(n_bath) * sqrt(hybrid/2/datum::pi/dos);
	};

	uword nx = 100;
	vec xgrid = linspace(x0_mpt-0.5, x0_fil+0.5, nx);
	int local_nx = nx / nprocs;

	mat local_Gamma = zeros(sz_sub, local_nx);
	mat local_val_cis_sub = zeros(sz_sub, local_nx);

	mat Gamma, val_cis_sub;
	if (id == 0) {
		Gamma.zeros(sz_sub, nx);
		val_cis_sub.zeros(sz_sub, nx);
		start = iclock::now();
	}

	TwoPara model(E_mpt, E_fil, bath, cpl, n_occ);

	for (int i = 0; i != local_nx; ++i) {
		model.calc( xgrid(id*local_nx+i) );
		local_val_cis_sub.col(i) = model.val_cis_sub;
		local_Gamma.col(i) = model.Gamma;
		std::cout << id*local_nx+i << "/" << nx << " finished" << std::endl;
	}

	::MPI_Gather(local_val_cis_sub.memptr(), local_val_cis_sub.n_elem, MPI_DOUBLE, val_cis_sub.memptr(), local_val_cis_sub.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	::MPI_Gather(local_Gamma.memptr(), local_Gamma.n_elem, MPI_DOUBLE, Gamma.memptr(), local_Gamma.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (id == 0) {
		cmd = "mkdir -p " + datadir;
		system_cmd = cmd.c_str();
		std::system(system_cmd);

		Gamma.save(datadir+"Gamma.txt", arma::raw_ascii);
		val_cis_sub.save(datadir+"val_cis_sub.txt", arma::raw_ascii);

		dur = iclock::now() - start;
		std::cout << "time elapsed = " << dur.count() << std::endl;
	}

	::MPI_Finalize();
	
	return 0;
}
