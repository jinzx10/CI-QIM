#include <mpi.h>
#include "mpi_helper.h"
#include "widgets.h"
#include "TwoPara2.h"
#include "arma_helper.h"

using namespace arma;

int main(int, char** argv) {

	int id, nprocs;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	Stopwatch sw;

	////////////////////////////////////////////////////////////
	//					Read-in Stage
	////////////////////////////////////////////////////////////
	// parameters to read from the command line
	std::string readdir, savedir;
	vec E0, E1, F0, F1, dc01, Gamma, xgrid;
	uword sz;

	if (id == 0) {
		readargs(argv, readdir, savedir);
		arma_load( readdir, 
				xgrid, "xgrid.dat",
				E0, "E0.dat",
				E1, "E1.dat",
				F0, "F0.dat",
				F1, "F1.dat",
				dc01, "dc01x.dat", // 'x' stands for approximation
				Gamma, "Gamma.dat"
		);
		sz = xgrid.n_elem;
		std::cout << "read successful" << std::endl;
		sw.run();
	}

	bcast(sz);

	if (id != 0) {
		set_size(sz, xgrid, E0, E1, F0, F1, dc01, Gamma);
	}

	bcast(xgrid, E0, E1, F0, F1, dc01, Gamma);

	TwoPara2 model(xgrid, E0, E1, F0, F1, arma::abs(dc01), Gamma);

	uword nx = 2000;
	vec x_fine = linspace(xgrid.min(), xgrid.max(), nx);
	vec E0_fine, E1_fine, F0_fine, F1_fine, dc01_fine, Gamma_fine;

	int nx_local = nx / nprocs;
	int rem = nx %  nprocs;
	if (id < rem)
		nx_local += 1;

	vec xgrid_local, E0_local, E1_local, F0_local, F1_local, 
		dc01_local, Gamma_local;
	set_size( nx_local, xgrid_local, E0_local, E1_local, F0_local, F1_local, 
		dc01_local, Gamma_local );

	int idx_start = ( nx / nprocs ) * id + ( id >= rem ? rem : id );
	for (int i = 0; i != nx_local; ++i) {
		double x = x_fine(idx_start+i);
		E0_local(i) = model.E0(x);
		E1_local(i) = model.E1(x);
		F0_local(i) = model.F0(x);
		F1_local(i) = model.F1(x);
		dc01_local(i) = model.dc01(x);
		Gamma_local(i) = model.Gamma(x);
		std::cout << idx_start+i+1 << "/" << nx << " finished" << std::endl;
	}

	if (id == 0) {
		set_size(nx, E0_fine, E1_fine, F0_fine, F1_fine, dc01_fine, Gamma_fine);
	}

	gatherv( E0_local, E0_fine, E1_local, E1_fine,
			F0_local, F0_fine, F1_local, F1_fine,
			dc01_local, dc01_fine, Gamma_local, Gamma_fine );

	if (id == 0) {
		mkdir(savedir);
		arma_save<raw_binary>( savedir,
				x_fine, "x_fine.txt",
				E0_fine, "E0_fine.txt",
				E1_fine, "E1_fine.txt",
				F0_fine, "F0_fine.txt",
				F1_fine, "F1_fine.txt",
				dc01_fine, "dc01_fine.txt",
				Gamma_fine, "Gamma_fine.txt"
		);
		sw.report();
	}

	MPI_Finalize();

	return 0;
}
