#include <mpi.h>
#include <string>
#include "mpi_helper.h"
#include "widgets.h"
#include "math_helper.h"
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
	uword n_bath = 0;
	double hybrid = 0.0;
	double doe_base = 0.0;
	double doe_peak = 0.0;
	double doe_width = 0.0;

	if (id == 0) {
		readargs(argv, datadir, n_bath, hybrid, doe_base, doe_peak, doe_width);
		std::cout << "data will be saved to: " << datadir << std::endl;
		std::cout << "number of bath states: " << n_bath << std::endl;
		std::cout << "hybridization Gamma: " << hybrid << std::endl;
		std::cout << "xgrid density: " << "base = " << doe_base 
			<< "   peak = " << doe_peak << "   width = " << doe_width 
			<< std::endl;
	}
	bcast(n_bath, hybrid, doe_base, doe_peak, doe_width);

	////////////////////////////////////////////////////////////
	//				Anderson Impurity Model
	////////////////////////////////////////////////////////////
	double W = 0.2;
	double bath_min = -W;
	double bath_max = W;
	vec bath = linspace<vec>(bath_min, bath_max, n_bath);
	vec dos = 1.0 / diff(bath);
	dos.insert_rows(dos.n_elem, 1);
	dos(dos.n_elem-1) = dos(dos.n_elem-2);

	double U = 0.1;
	uword n_occ = n_bath / 2;
	uword n_vir = n_bath + 1 - n_occ;
	double Ed1 = 0.0;
	double Ed2 = Ed1 - U;

	vec cpl = sqrt(hybrid/2.0/datum::pi/dos);

	// unevenly-spaced x grid; more samplings around the crossing point
	auto density = [&] (double E) { 
		return doe_base + 
			doe_peak * exp( -(E-Ed1)*(E-Ed1) / 2.0 / doe_width / doe_width ) + 
			doe_peak * exp( -(E-Ed2)*(E-Ed2) / 2.0 / doe_width / doe_width );
	};
	vec Edgrid = grid(-20, 40, density);
	uword nEd = Edgrid.n_elem;

	int nEd_local = nEd / nprocs;
	int rem = nEd % nprocs;
	if (id < rem)
		nEd_local += 1;

	uword sz_cis1d = n_occ + n_vir + 1;
	uword sz_cisnd = 2 * (n_occ + n_vir) - 1;

	// local variables and their initialization
	vec E_mf_local, n_mf_local;
	mat E_cis1d_local, n_cis1d_local;
	mat E_cisnd_local, n_cisnd_local;

	set_size(nEd_local, E_mf_local, n_mf_local);
	set_size(sz_cis1d, nEd_local, E_cis1d_local, n_cis1d_local);
	set_size(sz_cisnd, nEd_local, E_cisnd_local, n_cisnd_local);

	// global variables (used by proc 0)
	vec E_mf, n_mf;
	mat E_cis1d, n_cis1d;
	mat E_cisnd, n_cisnd;

	if (id == 0) {
		set_size(nEd, E_mf, n_mf);
		set_size(sz_cis1d, nEd, E_cis1d, n_cis1d);
		set_size(sz_cisnd, nEd, E_cisnd, n_cisnd);
		std::cout << "number of Ed grid points: " << nEd << std::endl;
		sw.run(0);
	}

	int idx_start = ( nEd / nprocs ) * id + ( id >= rem ? rem : id );

	// Single-impurity Anderson model
	SIAM model(bath, cpl, U, n_occ);

	for (int i = 0; i != nEd_local; ++i) {
		double Ed = Edgrid(idx_start+i);
		model.set_and_calc(Ed);
		
		E_mf_local(i) = model.E_mf;
		n_mf_local(i) = model.n_mf;

		if (id == 0)
			sw.report();

		std::cout << "proc id = " << id 
			<< "   local task: " << (i+1) << "/" << nEd_local << " finished"
			<< std::endl;
	}

	gatherv( n_mf_local, E_mf_local );

	if (id == 0) {
		mkdir(datadir);
		arma_save<raw_binary>( datadir,
				Edgrid, "Edgrid.dat", 
				E_mf, "E_mf.dat",
				n_mf, "n_mf.dat"
		);
		sw.report("program end");
		std::cout << std::endl << std::endl << std::endl;
	}

	MPI_Finalize();

	return 0;
}
