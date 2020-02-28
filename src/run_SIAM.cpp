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
	//				Anderson Impurity Model
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
	auto Ed = [&] (double const& x) { return E_fil(x) - E_mpt(x); };

	double W = 0.2;
	double bath_min = -W;
	double bath_max = W;
	vec bath = linspace<vec>(bath_min, bath_max, n_bath);
	vec dos = 1.0 / diff(bath);
	dos.insert_rows(dos.n_elem, 1);
	dos(dos.n_elem-1) = dos(dos.n_elem-2);

	double U = 0.1;
	double xU = 0;
	auto dE_U = [&] (double const& x) { return Ed(x) + U; };
	newtonroot(dE_U, xU);

	uword n_occ = n_bath / 2;
	uword n_vir = n_bath + 1 - n_occ;

	vec cpl = sqrt(hybrid/2.0/datum::pi/dos);

	// unevenly-spaced x grid; more samplings around the crossing point
	auto density = [&] (double x) { 
		return dox_base + 
			dox_peak * exp( -(x-xU)*(x-xU) / 2.0 / dox_width / dox_width ) + 
			dox_peak * exp( -(x-xc)*(x-xc) / 2.0 / dox_width / dox_width );
	};
	vec xgrid = grid(-20, 40, density);
	uword nx = xgrid.n_elem;

	int nx_local = nx / nprocs;
	int rem = nx % nprocs;
	if (id < rem)
		nx_local += 1;

	uword sz_cis1d = n_occ + n_vir + 1;
	uword sz_cisnd = 2 * (n_occ + n_vir) - 1;

	// local variables and their initialization
	vec E_mf_local, n_mf_local;
	mat E_cis1d_local, n_cis1d_local;
	mat E_cisnd_local, n_cisnd_local;

	set_size(nx_local, E_mf_local, n_mf_local);
	set_size(sz_cis1d, nx_local, E_cis1d_local, n_cis1d_local);
	set_size(sz_cisnd, nx_local, E_cisnd_local, n_cisnd_local);

	// global variables (used by proc 0)
	vec E_mf, n_mf;
	mat E_cis1d, n_cis1d;
	mat E_cisnd, n_cisnd;

	if (id == 0) {
		set_size(nx, E_mf, n_mf);
		set_size(sz_cis1d, nx, E_cis1d, n_cis1d);
		set_size(sz_cisnd, nx, E_cisnd, n_cisnd);
		std::cout << "number of Ed grid points: " << nx << std::endl;
		sw.run(0);
	}

	int idx_start = ( nx / nprocs ) * id + ( id >= rem ? rem : id );

	// Single-impurity Anderson model
	SIAM model(Ed, E_mpt, bath, cpl, U, n_occ);

	for (int i = 0; i != nx_local; ++i) {
		double x = xgrid(idx_start+i);
		model.set_and_calc(x);
		
		E_mf_local(i) = model.E_mf;
		n_mf_local(i) = model.n_mf;

		if (id == 0)
			sw.report();

		std::cout << "proc id = " << id 
			<< "   local task: " << (i+1) << "/" << nx_local << " finished"
			<< std::endl;
	}

	gatherv( n_mf_local, E_mf_local );

	if (id == 0) {
		mkdir(datadir);
		arma_save<raw_binary>( datadir,
				xgrid, "xgrid.dat", 
				E_mf, "E_mf.dat",
				n_mf, "n_mf.dat"
		);
		sw.report("program end");
		std::cout << std::endl << std::endl << std::endl;
	}

	MPI_Finalize();

	return 0;
}
