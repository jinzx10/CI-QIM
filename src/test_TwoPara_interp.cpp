#include <mpi.h>
#include <TwoPara_interp.h>
#include <chrono>
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

	std::string datadir = "/home/zuxin/job/CI-QIM/data/test_TwoPara/";

	vec E0, E1, F0, F1, dc01, Gamma, xgrid;
	uword sz;
	if (id == 0) {
		start = iclock::now();
		xgrid.load(datadir + "xgrid.txt");
		E0.load(datadir + "E0.txt");
		E1.load(datadir + "E1.txt");
		F0.load(datadir + "F0.txt");
		F1.load(datadir + "F1.txt");
		dc01.load(datadir + "dc01.txt");
		Gamma.load(datadir + "Gamma.txt");
		sz = xgrid.n_elem;
		std::cout << "read successful" << std::endl;
	}

	if (id == 0) {
		std::cout << sz << std::endl;
		std::cout << Gamma(0) << std::endl;
	}

	//::MPI_Bcast(&sz, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
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

	if (id == 0) {
		std::cout << "ready to bcast" << std::endl;
	}

	::MPI_Barrier(MPI_COMM_WORLD);
	bcast(xgrid, E0, E1, F0, F1, dc01, Gamma);
	::MPI_Barrier(MPI_COMM_WORLD);

	if (id == 0) {
		std::cout << "bcast successful" << std::endl;
	}

	TwoPara_interp model(xgrid, E0, E1, F0, F1, arma::abs(dc01), Gamma);
	if (id != 0) {
		std::cout << sz << std::endl;
		std::cout << Gamma(0) << std::endl;
	}

	uword nx = 2000;
	vec x_fine = linspace(xgrid.min(), xgrid.max(), nx);
	vec E0_fine, E1_fine, F0_fine, F1_fine, dc01_fine, Gamma_fine;

	uword nx_local = nx / nprocs;

	vec xgrid_local = zeros(nx_local);
	vec E0_local = zeros(nx_local);
	vec E1_local = zeros(nx_local);
	vec F0_local = zeros(nx_local);
	vec F1_local = zeros(nx_local);
	vec dc01_local = zeros(nx_local);
	vec Gamma_local = zeros(nx_local);

	for (uword i = 0; i != nx_local; ++i) {
		double x = x_fine(id*nx_local+i);
		E0_local(i) = model.E0(x);
		E1_local(i) = model.E1(x);
		F0_local(i) = model.F0(x);
		F1_local(i) = model.F1(x);
		dc01_local(i) = model.dc01(x);
		Gamma_local(i) = model.Gamma(x);
	}

	if (id == 0) {
		E0_fine.zeros(nx);
		E1_fine.zeros(nx);
		F0_fine.zeros(nx);
		F1_fine.zeros(nx);
		dc01_fine.zeros(nx);
		Gamma_fine.zeros(nx);
	}

	if (id == 0) {
		std::cout << "ready to gather" << std::endl;
	}

	::MPI_Barrier(MPI_COMM_WORLD);
	gather( E0_local, E0_fine, E1_local, E1_fine, 
			F0_local, F0_fine, F1_local, F1_fine,
			dc01_local, dc01_fine, Gamma_local, Gamma_fine );
	::MPI_Barrier(MPI_COMM_WORLD);

	if (id == 0) {
		std::cout << "gather successful" << std::endl;
	}

	if (id == 0) {
		x_fine.save(datadir + "x_fine.txt", arma::raw_ascii);
		E0_fine.save(datadir + "E0_fine.txt", arma::raw_ascii);
		E1_fine.save(datadir + "E1_fine.txt", arma::raw_ascii);
		F0_fine.save(datadir + "F0_fine.txt", arma::raw_ascii);
		F1_fine.save(datadir + "F1_fine.txt", arma::raw_ascii);
		dc01_fine.save(datadir + "dc01_fine.txt", arma::raw_ascii);
		Gamma_fine.save(datadir + "Gamma_fine.txt", arma::raw_ascii);
		dur = iclock::now() - start;
		std::cout << "time elapsed = " << dur.count() << std::endl;
	}

	MPI_Finalize();

	return 0;
}
