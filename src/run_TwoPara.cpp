#include <mpi.h>
#include <string>
#include "TwoPara.h"
#include "mpi_helper.h"
#include "arma_helper.h"
#include "math_helper.h"
#include "widgets.h"

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
	std::string datadir;
	uword n_bath = 0;
	double hybrid = 0.0;
	double dox_base = 0.0;
	double dox_peak = 0.0;
	double dox_width = 0.0;

	if (id == 0) {
		readargs(argv, datadir, n_bath, hybrid, dox_base, dox_peak, dox_width);
		std::cout << "data will be saved to: " << datadir << std::endl;
		std::cout << "number of bath states: " << n_bath << std::endl;
		std::cout << "hybridization Gamma: " << hybrid << std::endl;
		std::cout << "xgrid density: " << "base = " << dox_base 
			<< "   peak = " << dox_peak << "   width = " << dox_width 
			<< std::endl;
	}
	bcast(n_bath, hybrid, dox_base, dox_peak, dox_width);

	////////////////////////////////////////////////////////////
	//					Two-Parabola model
	////////////////////////////////////////////////////////////
	double x0_mpt = 0;
	double x0_fil = 20.6097;
	double omega = 0.0002;
	double mass = 2000;
	double dE_fil = -0.0038;
	double xc = 0.5*(x0_mpt+x0_fil) + dE_fil / ( mass*omega*omega*(x0_fil-x0_mpt) );

	auto E_mpt = [&] (double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_mpt) * (x - x0_mpt);};
	auto E_fil = [&] (double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_fil) * (x - x0_fil) + dE_fil;};

	double W = 0.1;
	double bath_min = -W;
	double bath_max = W;
	vec bath = linspace<vec>(bath_min, bath_max, n_bath);
	vec dos = 1.0 / diff(bath);
	dos.insert_rows(dos.n_elem, 1);
	dos(dos.n_elem-1) = dos(dos.n_elem-2);

	uword n_occ = n_bath / 2;

	vec cpl = sqrt(hybrid/2.0/datum::pi/dos);

	// unevenly-spaced x grid; more samplings around the crossing point
	auto density = [&] (double x) { 
		return dox_base + 
			dox_peak * exp( -(x-xc)*(x-xc) / 2.0 / dox_width / dox_width );
	};
	vec xgrid = grid(-20, 40, density);
	uword nx = xgrid.n_elem;

	int nx_local = nx / nprocs;
	int rem = nx % nprocs;
	if (id < rem)
		nx_local += 1;

	// local variables and their initialization
	vec E0_local, E1_local, F0_local, F1_local,
		dc01_local, dc01x_local, Gamma_local, n_imp_local;

	set_size( nx_local, E0_local, E1_local, F0_local, F1_local,
			dc01_local, dc01x_local, Gamma_local, n_imp_local );

	// global variables (used by proc 0)
	vec E0, E1, F0, F1, dc01, dc01x, Gamma, n_imp;

	if (id == 0) {
		set_size( nx, E0, E1, F0, F1, dc01, dc01x, Gamma, n_imp );
		std::cout << "number of x grid points: " << nx << std::endl;
		sw.run(0);
	}

	int idx_start = ( nx / nprocs ) * id + ( id >= rem ? rem : id );

	// Two parabola model
	double x_init = xgrid(idx_start) - 1e-3;
	TwoPara model(E_mpt, E_fil, bath, cpl, n_occ, x_init);

	for (int i = 0; i != nx_local; ++i) {
		double x = xgrid(idx_start+i);
		model.set_and_calc(x);
		
		E0_local(i) = model.ev_H + E_mpt(x);
		E1_local(i) = model.val_cis_sub(0) + E_mpt(x);
		F0_local(i) = model.force(0);
		F1_local(i) = model.force(1);
		dc01x_local(i) = model.dc(0,1);
		Gamma_local(i) = model.Gamma(0);
		n_imp_local(i) = model.ev_n;

		if (id == 0)
			sw.report(0);

		std::cout << "proc id = " << id 
			<< "   local task: " << (i+1) << "/" << nx_local << " finished"
			<< std::endl;
	}

	gatherv( E0_local, E0, E1_local, E1, F0_local, F0, F1_local, F1,
			dc01_local, dc01, dc01x_local, dc01x,
			Gamma_local, Gamma, n_imp_local, n_imp );

	if (id == 0) {
		mkdir(datadir);
		arma_save<raw_binary>( datadir,
				xgrid, "xgrid.dat", 
				E0, "E0.dat", 
				E1, "E1.dat",
				F0, "F0.dat",
				F1, "F1.dat",
				//dc01, "dc01.dat", 
				dc01x, "dc01x.dat", 
				Gamma, "Gamma.dat",
				n_imp, "n_imp.dat"
		);
		sw.report("total program");
		std::cout << "end of program" << std::endl;
		std::cout << std::endl << std::endl << std::endl;
	}

	MPI_Finalize();
	
	return 0;
}


