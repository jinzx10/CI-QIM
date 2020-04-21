#include <mpi.h>
#include <string>
#include "mpi_helper.h"
#include "widgets.h"
#include "math_helper.h"
#include "arma_helper.h"
#include "SIAM.h"

using namespace arma;

int main(int, char**argv) {
	
	int id, nprocs;

	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	Stopwatch sw;

	////////////////////////////////////////////////////////////
	//					Read-in Stage
	////////////////////////////////////////////////////////////
	std::string datadir;
	double bathdos_base = 0.0;
	double hybrid = 0.0;
	double dox_base = 0.0;
	double dox_peak = 0.0;
	double dox_width = 0.0;

	if (id == 0) {
		readargs(argv, datadir, bathdos_base, hybrid, 
				dox_base, dox_peak, dox_width);
		std::cout << "data will be saved to: " << datadir << std::endl;
		std::cout << "density of bath states: " << bathdos_base << std::endl;
		std::cout << "hybridization Gamma: " << hybrid << std::endl;
		std::cout << "xgrid density: " << "base = " << dox_base 
			<< "   peak = " << dox_peak << "   width = " << dox_width 
			<< std::endl;
	}
	bcast(bathdos_base, hybrid, dox_base, dox_peak, dox_width);

	////////////////////////////////////////////////////////////
	//				Anderson Impurity Model
	////////////////////////////////////////////////////////////
	// impurity
	double x0_mpt = 0;
	double x0_fil = 20.6097;
	double omega = 0.0004;
	double mass = 2000;
	double dE_fil = -0.0038;
	double U = 0.05;

	auto E_mpt = [&] (double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_mpt) * (x - x0_mpt);};
	auto E_fil = [&] (double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_fil) * (x - x0_fil) + dE_fil;};
	auto E_imp = [&] (double const& x) { return E_fil(x) - E_mpt(x); };
	auto E_imp2 = [&] (double const& x) { return E_imp(x) + U; };

	double xc = 0.0;
	double xc2 = 0.0;
	newtonroot(E_imp, xc);
	newtonroot(E_imp2, xc2);

	if (id == 0) {
		std::cout << "xc = " << xc << "        "
			<< "xc2 = " << xc2 << std::endl;
	}

	// bath
	double W = 0.2;
	double bath_min = -W;
	double bath_max = W;
	auto bathdos = [&] (double) { return bathdos_base; };
	vec bath = grid(bath_min, bath_max, bathdos);
	uword n_bath = bath.n_elem;
	vec dos = bath;
	dos.for_each([&](double& elem) {elem = bathdos(elem);});
	
	uword n_occ = n_bath / 2;
	uword n_vir = n_bath + 1 - n_occ;

	vec cpl = sqrt(hybrid/2.0/datum::pi/dos);

	if (id == 0) {
		std::cout << "number of bath states = " << n_bath << std::endl;
	}

	// x grid
	auto density = [&] (double x) { 
		return dox_base + 
			dox_peak * exp( -(x-xc)*(x-xc) / 2.0 / dox_width / dox_width ) + 
			dox_peak * exp( -(x-xc2)*(x-xc2) / 2.0 / dox_width / dox_width );
	};
	vec xgrid = grid(-20, 40, density);
	uword nx = xgrid.n_elem;

	if (id == 0) {
		std::cout << "number of x grid points: " << nx << std::endl;
	}

	int nx_local = nx / nprocs;
	int rem = nx % nprocs;
	if (id < rem)
		nx_local += 1;

	uword sz_cisnd = 2 * (n_occ + n_vir) - 1;

	// local variables and their initialization
	vec E_mf_local, n_mf_local;
	mat E_cisnd_local, n_cisnd_local;

	set_size(nx_local, E_mf_local, n_mf_local);
	set_size(sz_cisnd, nx_local, E_cisnd_local, n_cisnd_local);

	// global variables (used by proc 0)
	vec E_mf, n_mf;
	mat E_cisnd, n_cisnd;

	if (id == 0) {
		set_size(nx, E_mf, n_mf);
		set_size(sz_cisnd, nx, E_cisnd, n_cisnd);
		sw.run(0);
	}

	int idx_start = ( nx / nprocs ) * id + ( id >= rem ? rem : id );

	// model initialization
	double x0 = xgrid(idx_start) - 0.001;
	uword sz_sub = 30;
	SIAM model(E_imp, E_mpt, bath, cpl, U, n_occ, sz_sub, x0);

	if (id == 0) {
		std::cout << "model initialized" << std::endl;
	}

	for (int i = 0; i != nx_local; ++i) {
		double x = xgrid(idx_start+i);
		model.set_and_calc(x);
		
		E_mf_local(i) = model.E_mf;
		n_mf_local(i) = model.n_mf;
		E_cisnd_local.col(i) = model.val_cisnd;
		n_cisnd_local.col(i) = model.n_cisnd;

		if (id == 0)
			sw.report();

		std::cout << "proc id = " << id 
			<< "   local task: " << (i+1) << "/" << nx_local << " finished"
			<< std::endl;
	}

	gatherv( n_mf_local, n_mf, E_mf_local, E_mf, E_cisnd_local, 
			 E_cisnd, n_cisnd_local, n_cisnd );

	if (id == 0) {
		mkdir(datadir);
		arma_save<raw_binary>( datadir,
				xgrid, "xgrid.dat", 
				E_mf, "E_mf.dat",
				n_mf, "n_mf.dat",
				E_cisnd, "E_cisnd.dat",
				n_cisnd, "n_cisnd.dat"
		);
		sw.report("program end");
		std::cout << std::endl << std::endl << std::endl;
	}

	MPI_Finalize();

	return 0;
}
