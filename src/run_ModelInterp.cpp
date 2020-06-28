#include <mpi.h>
#include "mpi_helper.h"
#include "widgets.h"
#include "ModelInterp.h"
#include "arma_helper.h"

using namespace arma;

int main(int, char** argv) {

	int id, nprocs, root = 0;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	Stopwatch sw;

	////////////////////////////////////////////////////////////
	//					Read-in Stage
	////////////////////////////////////////////////////////////
	std::string readdir, savedir;
	vec xgrid;
	mat pes, dc, Gamma, force, n_imp;
	uword sz_x, sz_elec;

	if (id == root) {
		readargs(argv, readdir, savedir);

		readdir = expand_leading_tilde(readdir);
		savedir = expand_leading_tilde(savedir);

		std::cout << "read data from " << readdir << std::endl;
		std::cout << "save data to " << savedir << std::endl;

		arma_load( readdir, 
				xgrid, "xgrid.dat",
				pes, "E_adi.dat",
				n_imp, "n_imp.dat",
				dc, "dc_adi.dat",
				force, "F_adi.dat",
				Gamma, "Gamma_rlx.dat"
		);
		sz_x = xgrid.n_elem;
		sz_elec = pes.n_elem / sz_x;

		pes.reshape(sz_elec, sz_x);
		n_imp.reshape(sz_elec, sz_x);
		force.reshape(sz_elec, sz_x);
		Gamma.reshape(sz_elec, sz_x);
		dc.reshape(sz_elec*sz_elec, sz_x);

		std::cout << "data read successfully" << std::endl;
		std::cout << "# of coarse xgrid points: " << sz_x << std::endl;
		std::cout << "size of electronic basis: " << sz_elec << std::endl;
	}

	bcast(root, sz_x, sz_elec);

	if (id != root) {
		set_size(sz_x, xgrid);
		set_size(sz_elec, sz_x, pes, n_imp, force, Gamma);
		set_size(sz_elec*sz_elec, sz_x, dc);
	}

	bcast(root, xgrid, pes, n_imp, dc, force, Gamma);

	ModelInterp model(xgrid, pes.t(), n_imp.t(), force.t(), dc.t(), Gamma.t());

	uword nx = 2000;
	vec x_fine = linspace(xgrid.min(), xgrid.max(), nx);
	mat pes_fine, dc_fine, Gamma_fine, force_fine, n_imp_fine;

	if (id == root) {
		std::cout << "model initialized" << std::endl;
		set_size(sz_elec, nx, pes_fine, n_imp_fine, force_fine, Gamma_fine);
		set_size(sz_elec*sz_elec, nx, dc_fine);
		sw.run();
	}

	int nx_local = nx / nprocs;
	int rem = nx % nprocs;
	if (id < rem)
		nx_local += 1;

	mat pes_local, dc_local, force_local, Gamma_local, n_imp_local;
	set_size(sz_elec, nx_local, pes_local, force_local, Gamma_local, n_imp_local);
	set_size(sz_elec*sz_elec, nx_local, dc_local);

	int idx_start = ( nx / nprocs ) * id + ( id >= rem ? rem : id );
	for (int i = 0; i != nx_local; ++i) {
		double x = x_fine(idx_start+i);
		pes_local.col(i) = model.E(x);
		n_imp_local.col(i) = model.n_imp(x);
		force_local.col(i) = model.F(x);
		Gamma_local.col(i) = model.Gamma(x);
		dc_local.col(i) = model.dc(x).as_col();

		if (nprocs == 1) {
			if (i == 0)
				std::cout << std::endl << std::endl;
			std::cout << "\033[A\033[2K\033[A\033[2K\r";
		}

		if (id == root)
			sw.report();

		std::cout << "proc id = " << id 
			<< "   local task: " << (i+1) << "/" << nx_local << " finished"
			<< std::endl;
	}

	gatherv( root, pes_local, pes_fine, n_imp_local, n_imp_fine, 
			force_local, force_fine, Gamma_local, Gamma_fine, dc_local, dc_fine 
	);

	if (id == root) {
		mkdir(savedir);
		arma_save<raw_binary>( savedir,
				x_fine, "x_fine.dat",
				pes_fine, "E_fine.dat",
				n_imp_fine, "n_imp_fine.dat",
				force_fine, "F_fine.dat",
				dc_fine, "dc_fine.dat",
				Gamma_fine, "Gamma_fine.dat"
		);
		sw.report("total program");
	}

	MPI_Finalize();

	return 0;
}
