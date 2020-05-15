#include <mpi.h>
#include <string>
#include <stdlib.h>
#include "TwoPara.h"
#include "mpi_helper.h"
#include "arma_helper.h"
#include "math_helper.h"
#include "widgets.h"

using namespace arma;

int main(int, char** argv) {

	int id, nprocs;
	int root = 0;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	Stopwatch sw;

	////////////////////////////////////////////////////////////
	//					Read-in Stage
	////////////////////////////////////////////////////////////
	std::string file_path;
	Parser p({"savedir", "x0_mpt", "x0_fil", "omega", "mass", "dE_fil", 
			"W", "dos_base", "hybrid", "dox_base", "dox_peak", "dox_width", 
			"sz_sub", "left_wall", "right_wall"});

	std::string savedir;
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
	double W = 0.0;
	uword sz_sub = 0;
	double left_wall = 0.0;
	double right_wall = 0.0;

	if (id == root) {
		readargs(argv, file_path);
		p.parse(file_path);
		p.pour(savedir, x0_mpt, x0_fil, omega, mass, dE_fil,
				W, dos_base, hybrid, dox_base, dox_peak, dox_width, 
				sz_sub, left_wall, right_wall);

		std::cout << savedir << std::endl;

		savedir = expand_leading_tilde(savedir);

		std::cout << "data will be saved to: " << savedir << std::endl
			<< "x0_mpt = " << x0_mpt << std::endl
			<< "x0_fil = " << x0_fil << std::endl
			<< "omega = " << omega << std::endl
			<< "mass = " << mass << std::endl
			<< "dE_fil = " << dE_fil << std::endl
			<< "W = " << W << std::endl
			<< "bath density of states = " << dos_base << std::endl
			<< "hybridization = " << hybrid << std::endl
			<< "base density of x grid = " << dox_base << std::endl
			<< "peak density of x grid = " << dox_peak << std::endl
			<< "width of x grid peak density = " << dox_width << std::endl
			<< "size of subspace adiabats = " << sz_sub << std::endl
			<< std::endl;
	}

	bcast(root, x0_mpt, x0_fil, omega, mass, dE_fil,
			W, dos_base, hybrid, dox_base, dox_peak, dox_width, 
			sz_sub, left_wall, right_wall);

	////////////////////////////////////////////////////////////
	//					Two-Parabola model
	////////////////////////////////////////////////////////////
	// impurity
	auto E_mpt = [&] (double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_mpt) * (x - x0_mpt);};
	auto E_fil = [&] (double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_fil) * (x - x0_fil) + dE_fil;};
	auto E_imp = [&] (double const& x) { return E_fil(x) - E_mpt(x); };

	double xc = 0.0;
	newtonroot(E_imp, xc);


	// bath
	auto bathdos = [&] (double) { return dos_base; };
	vec bath = grid(-W, W, bathdos);
	uword n_bath = bath.n_elem;
	vec dos = bath;
	dos.for_each([&](double& elem) {elem = bathdos(elem);});
	
	uword n_occ = n_bath / 2;
	uword n_vir = n_bath + 1 - n_occ;
	uword sz_cis = n_occ + n_vir - 1;

	vec cpl = sqrt(hybrid/2.0/datum::pi/dos);

	// x grid; more samplings around the crossing point
	auto density = [&] (double x) { 
		return dox_base + dox_peak * gauss(x, xc, dox_width);
	};

	vec xgrid = grid(x0_mpt-left_wall*(xc-x0_mpt), x0_fil+right_wall*(x0_fil-xc), density);
	uword nx = xgrid.n_elem;

	uword nx_local = nx / nprocs;
	int rem = nx % nprocs;
	if (id < rem)
		nx_local += 1;

	if (id == root) {
		std::cout << "diabatic crossing = " << xc << std::endl
			<< "number of bath states = " << n_bath << std::endl 
			<< "size of selective CIS basis = " << sz_cis << std::endl 
			<< "number of x grid points: " << nx << std::endl
			<< "number of x grid points for proc-0: " << nx_local << std::endl
			<< "minimum on-site energy: " << E_imp(xgrid(nx-1)) << std::endl
			<< "maximum on-site energy: " << E_imp(xgrid(0)) << std::endl
			<< std::endl;
	}

	// local variables and their initialization
	vec n_imp_local;
	mat E_adi_local, F_adi_local, dc_adi_local, Gamma_rlx_local;

	set_size(nx_local, n_imp_local);
	set_size({sz_sub, nx_local}, E_adi_local, F_adi_local, Gamma_rlx_local);
	set_size({sz_sub*sz_sub, nx_local}, dc_adi_local);

	// global variables (used by proc 0)
	vec n_imp;
	mat E_adi, F_adi, dc_adi, Gamma_rlx;

	int idx_start = ( nx / nprocs ) * id + ( id >= rem ? rem : id );

	// Two parabola model
	double x_init = xgrid(idx_start) - 1e-3;
	TwoPara model(E_mpt, E_fil, bath, cpl, n_occ, sz_sub, x_init);

	if (id == root) {
		set_size(nx, n_imp);
		set_size({sz_sub, nx}, E_adi, F_adi, Gamma_rlx);
		set_size({sz_sub*sz_sub, nx}, dc_adi);
		sw.run(0);
		std::cout << "model initialized" << std::endl;
	}


	for (uword i = 0; i != nx_local; ++i) {
		double x = xgrid(idx_start+i);
		model.set_and_calc(x);
		
		E_adi_local.col(i) = model.E_sub();
		F_adi_local.col(i) = model.F_sub();
		dc_adi_local.col(i) = model.dc_adi.as_col();
		Gamma_rlx_local.col(i) = model.Gamma_rlx;
		n_imp_local(i) = model.ev_n;

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

	gatherv( E_adi_local, E_adi, F_adi_local, F_adi, Gamma_rlx_local, Gamma_rlx, 
			dc_adi_local, dc_adi, n_imp_local, n_imp);

	std::fstream fs;
	std::string paramfile;
	if (id == root) {
		mkdir(savedir);
		arma_save<raw_binary>( savedir,
				xgrid, "xgrid.dat", 
				E_adi, "E_adi.dat", 
				F_adi, "F_adi.dat",
				dc_adi, "dc_adi.dat",
				Gamma_rlx, "Gamma_rlx.dat",
				n_imp, "n_imp.dat"
		);

		paramfile = savedir+"/param.txt";
		touch(paramfile);

		fs.open(paramfile);
		fs << "omega " << omega << std::endl
			<< "mass " << mass << std::endl
			<< "x0_mpt " << x0_mpt << std::endl;
		fs.close();

		sw.report("total program");
		std::cout << std::endl << std::endl << std::endl;
	}

	MPI_Finalize();
	
	return 0;
}


