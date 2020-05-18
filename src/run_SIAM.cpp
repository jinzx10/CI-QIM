#include <mpi.h>
#include <string>
#include <fstream>
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
	std::string file_path;
	Parser p({"datadir", "x0_mpt", "x0_fil", "omega", "mass", "dE_fil", "U", 
			"W", "dos_base", "hybrid", "dox_base", "dox_peak", "dox_width", 
			"sz_sub"});
	
	std::string datadir;
	double dos_base = 0.0;
	double hybrid = 0.0;
	double dox_base = 0.0;
	double dox_peak = 0.0;
	double dox_width = 0.0;
	double x0_mpt = 0.0;
	double x0_fil = 0.0;
	double omega = 0.0;
	double mass = 0.0;
	double dE_fil = 0.0;
	double U = 0.0;
	double W = 0.0;
	uword sz_sub = 0;

	if (id == 0) {
		readargs(argv, file_path);
		p.parse(file_path);
		p.pour(datadir, x0_mpt, x0_fil, omega, mass, dE_fil, U, 
				W, dos_base, hybrid, dox_base, dox_peak, dox_width, sz_sub);

		datadir = expand_leading_tilde(datadir);

		std::cout << "data will be saved to: " << datadir << std::endl
			<< "x0_mpt = " << x0_mpt << std::endl
			<< "x0_fil = " << x0_fil << std::endl
			<< "omega = " << omega << std::endl
			<< "mass = " << mass << std::endl
			<< "dE_fil = " << dE_fil << std::endl
			<< "U = " << U << std::endl
			<< "W = " << W << std::endl
			<< "bath density of states = " << dos_base << std::endl
			<< "hybridization = " << hybrid << std::endl
			<< "base density of x grid = " << dox_base << std::endl
			<< "peak density of x grid = " << dox_peak << std::endl
			<< "width of x grid peak density = " << dox_width << std::endl
			<< "size of subspace adiabats = " << sz_sub << std::endl
			<< std::endl;
	}

	bcast(x0_mpt, x0_fil, omega, mass, dE_fil, U, 
			W, dos_base, hybrid, dox_base, dox_peak, dox_width, sz_sub);

	////////////////////////////////////////////////////////////
	//				Anderson Impurity Model
	////////////////////////////////////////////////////////////
	// impurity
	auto E_mpt = [&] (double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_mpt) * (x - x0_mpt);};
	auto E_fil = [&] (double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_fil) * (x - x0_fil) + dE_fil;};
	auto E_imp = [&] (double const& x) { return E_fil(x) - E_mpt(x); };
	auto E_imp2 = [&] (double const& x) { return E_imp(x) + U; };

	double x0_fil2 = 2.0*x0_fil - x0_mpt;
	double xc = 0.0;
	double xc2 = 0.0;
	broydenroot(E_imp, xc);
	broydenroot(E_imp2, xc2);

	// bath
	auto bathdos = [&] (double) { return dos_base; };
	vec bath = grid(-W, W, bathdos);
	uword n_bath = bath.n_elem;
	vec dos = bath;
	dos.for_each([&](double& elem) {elem = bathdos(elem);});
	
	uword n_occ = n_bath / 2;
	uword n_vir = n_bath + 1 - n_occ;
	uword sz_cisnd = 2 * (n_occ + n_vir) - 1;

	vec cpl = sqrt(hybrid/2.0/datum::pi/dos);

	// x grid
	auto density = [&] (double x) { 
		return dox_base + dox_peak * ( gauss(x, xc, dox_width) + 
				gauss(x, xc2, dox_width) );
	};
	vec xgrid = grid(2.0*x0_mpt-xc, 2.0*x0_fil2-xc2, density);
	uword nx = xgrid.n_elem;

	int nx_local = nx / nprocs;
	int rem = nx % nprocs;
	if (id < rem)
		nx_local += 1;

	if (id == 0) {
		std::cout << "0-1 diabatic crossing = " << xc << std::endl
			<< "1-2 diabatic crossing = " << xc2 << std::endl
			<<"number of bath states = " << n_bath << std::endl 
			<< "size of selective CISND basis = " << sz_cisnd << std::endl 
			<< "number of x grid points: " << nx << std::endl
			<< "number of x grid points for proc-0: " << nx_local << std::endl
			<< std::endl;
	}

	// local variables and their initialization
	vec E_mf_local, n_mf_local;
	mat E_adi_local, n_cisnd_local, F_adi_local, Gamma_rlx_local;
	mat dc_adi_local;

	set_size(nx_local, E_mf_local, n_mf_local);
	set_size(sz_sub, nx_local, E_adi_local, n_cisnd_local, 
			F_adi_local, Gamma_rlx_local);
	set_size(sz_sub*sz_sub, nx_local, dc_adi_local);

	// global variables (used by proc 0)
	vec E_mf, n_mf;
	mat E_adi, n_cisnd, F_adi;
	mat dc_adi;
	mat Gamma_rlx;

	int idx_start = ( nx / nprocs ) * id + ( id >= rem ? rem : id );

	// model initialization
	double x0 = xgrid(idx_start) - 0.001;
	SIAM model(E_imp, E_mpt, bath, cpl, U, n_occ, sz_sub, x0);

	if (id == 0) {
		set_size(nx, E_mf, n_mf);
		set_size(sz_sub, nx, E_adi, n_cisnd, F_adi, Gamma_rlx);
		set_size(sz_sub*sz_sub, nx, dc_adi);
		sw.run(0);
		std::cout << "model initialized" << std::endl;
	}

	for (int i = 0; i != nx_local; ++i) {
		double x = xgrid(idx_start+i);
		model.set_and_calc(x);
		
		E_mf_local(i) = model.E_mf;
		n_mf_local(i) = model.n_mf;
		E_adi_local.col(i) = model.val_cisnd + model.E_nuc(x);
		n_cisnd_local.col(i) = model.n_cisnd;
		F_adi_local.col(i) = model.F_cisnd + model.F_nucl;
		dc_adi_local.col(i) = model.dc_adi.as_col();
		Gamma_rlx_local.col(i) = model.Gamma_rlx;

		if (nprocs == 1) {
			if (i == 0)
				std::cout << std::endl << std::endl;
			std::cout << "\033[A\033[2K\033[A\033[2K\r";
		}

		if (id == 0)
			sw.report();

		std::cout << "proc id = " << id 
			<< "   local task: " << (i+1) << "/" << nx_local << " finished"
			<< std::endl;
	}

	gatherv( n_mf_local, n_mf, E_mf_local, E_mf, E_adi_local, E_adi, 
			n_cisnd_local, n_cisnd, F_adi_local, F_adi,
			dc_adi_local, dc_adi, Gamma_rlx_local, Gamma_rlx );

	std::fstream fs;
	std::string paramfile;
	if (id == 0) {
		mkdir(datadir);
		arma_save<raw_binary>( datadir,
				xgrid, "xgrid.dat", 
				E_mf, "E_mf.dat",
				n_mf, "n_mf.dat",
				E_adi, "E_adi.dat",
				n_cisnd, "n_cisnd.dat",
				F_adi, "F_adi.dat",
				dc_adi, "dc_adi.dat",
				Gamma_rlx, "Gamma_rlx.dat"
		);

		paramfile = datadir+"param.txt";
		touch(paramfile);

		fs.open(paramfile);
		fs << "omega " << omega << std::endl
			<< "mass " << mass << std::endl
			<< "x0_mpt " << x0_mpt << std::endl;
		fs.close();

		sw.report("program end");
		std::cout << std::endl << std::endl << std::endl; 
	}

	MPI_Finalize();

	return 0;
}
