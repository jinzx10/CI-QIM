#include <TwoPara.h>
#include <chrono>
#include <mpi.h>
#include <cstdlib>
#include <arma_mpi_helper.h>

using namespace arma;
using iclock = std::chrono::high_resolution_clock;

int main() {

	int id, nprocs;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	iclock::time_point start;
	std::chrono::duration<double> dur;
	std::string datadir = "/home/zuxin/job/CI-QIM/data/test_TwoPara/Gamma/00128_1000/";
	//std::string datadir = "/home/zuxin/job/CI-QIM/data/test_TwoPara/";
	std::string cmd;
	const char* system_cmd = nullptr;
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

	uword nx = 32*3*10;
	vec xgrid = linspace(-20, 40, nx);
	int local_nx = nx / nprocs;

	// global
	vec E0, E1, F0, F1, dc01, Gamma, n_imp;

	// local
	vec local_E0 = zeros(local_nx);
	vec local_E1 = zeros(local_nx);
	vec local_F0 = zeros(local_nx);
	vec local_F1 = zeros(local_nx);
	vec local_dc01 = zeros(local_nx);
	vec local_Gamma = zeros(local_nx);
	vec local_n_imp = zeros(local_nx);

	uword sz_dc = 2;

	if (id == 0) {
		E0.zeros(nx);
		E1.zeros(nx);
		F0.zeros(nx);
		F1.zeros(nx);
		dc01.zeros(nx);
		Gamma.zeros(nx);
		n_imp.zeros(nx);

		start = iclock::now();
	}

	TwoPara model(E_mpt, E_fil, bath, cpl, n_occ);

	for (int i = 0; i != local_nx; ++i) {
		double x = xgrid(id*local_nx+i);
		model.set_and_calc_cis_sub(x);
		model.calc_val_cis_bath();
		model.calc_Gamma(1);

		local_E0(i) = model.ev_H + E_mpt(x);
		local_E1(i) = model.val_cis_sub(0) + E_mpt(x);
		local_F0(i) = model.force(0);
		local_F1(i) = model.force(1);
		local_dc01(i) = model.dc(sz_dc)(0,1);
		local_Gamma(i) = model.Gamma(0);
		local_n_imp(i) = model.ev_n;

		std::cout << id*local_nx+i+1 << "/" << nx << " finished" << std::endl;
	}

	::gather(local_E0, E0, local_E1, E1, local_F0, F0, local_F1, F1, 
			local_dc01, dc01, local_Gamma, Gamma, local_n_imp, n_imp);

	::MPI_Barrier(MPI_COMM_WORLD);

	if (id == 0) {
		cmd = "mkdir -p " + datadir;
		system_cmd = cmd.c_str();
		status = std::system(system_cmd);

		xgrid.save(datadir+"xgrid.txt", arma::raw_ascii);
		E0.save(datadir+"E0.txt", arma::raw_ascii);
		E1.save(datadir+"E1.txt", arma::raw_ascii);
		F0.save(datadir+"F0.txt", arma::raw_ascii);
		F1.save(datadir+"F1.txt", arma::raw_ascii);
		dc01.save(datadir+"dc01.txt", arma::raw_ascii);
		Gamma.save(datadir+"Gamma.txt", arma::raw_ascii);
		n_imp.save(datadir+"n_imp.txt", arma::raw_ascii);

		dur = iclock::now() - start;
		std::cout << "time elapsed = " << dur.count() << std::endl;
	}

	::MPI_Barrier(MPI_COMM_WORLD);
	::MPI_Finalize();
	
	return 0;
}
