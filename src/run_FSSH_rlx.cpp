#include <mpi.h>
#include "FSSH_rlx.h"
#include "ModelInterp.h"
#include "mpi_helper.h"
#include "arma_helper.h"
#include "widgets.h"
#include "math_helper.h"

using namespace arma;

int main(int, char**argv) {

	int id, nprocs, root = 0;

	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	Stopwatch sw;

	////////////////////////////////////////////////////////////
	//					Read-in Stage
	////////////////////////////////////////////////////////////
	std::string input_file;
	Parser p({
			"readdir", 
			"savedir", 
			"n_trajs", 
			"t_max", 
			"dtc", 
			"velo_rev", 
			"velo_rescale", 
			"fric_mode", 
			"kT", 
			"sz_elec", 
			"has_rlx",
			"n_only"
	});

	std::string readdir;
	std::string savedir;
	uword n_trajs;
	double t_max;
	double dtc;
	int velo_rev;
	int velo_rescale;
	int has_rlx;
	int n_only;
	int fric_mode;
	double kT;
	uword sz_elec_fssh; // used in FSSH

	double omega;
	double mass;
	double x0_mpt;
	
	vec xgrid;
	mat pes, n_imp, force, Gamma, dc;
	uword sz_x, sz_elec; // sz_elec of data

	std::string paramfile;
	if (id == root) {
		readargs(argv, input_file);

		p.parse(input_file);
		p.pour(readdir, savedir, n_trajs, t_max, dtc, 
				velo_rev, velo_rescale, fric_mode, kT, sz_elec_fssh, has_rlx, n_only);

		readdir = expand_leading_tilde(readdir);
		savedir = expand_leading_tilde(savedir);

		paramfile = readdir + "/param.txt";
		p.reset({"omega", "mass", "x0_mpt"});
		p.parse(paramfile);
		p.pour(omega, mass, x0_mpt);

		arma_load( readdir, 
				xgrid, "xgrid.dat",
				pes, "E_adi.dat",
				n_imp, "n_imp.dat",
				force, "F_adi.dat",
				Gamma, "Gamma_rlx.dat",
				dc, "dc_adi.dat"
		);

		sz_x = xgrid.n_elem;
		sz_elec = pes.n_elem / sz_x;

		pes.reshape(sz_elec, sz_x);
		n_imp.reshape(sz_elec, sz_x);
		force.reshape(sz_elec, sz_x);
		Gamma.reshape(sz_elec, sz_x);
		dc.reshape(sz_elec*sz_elec, sz_x);

	}

	bcast(root, n_trajs, t_max, dtc, velo_rev, velo_rescale, fric_mode, kT, 
			omega, mass, x0_mpt, sz_x, sz_elec, sz_elec_fssh, has_rlx, n_only);

	if (id == nprocs-1) {
		std::cout << "data are read from: " << readdir << std::endl
			<< "data will be saved to: " << savedir << std::endl
			<< "# of trajectories = " << n_trajs << std::endl
			<< "max time = " << t_max << std::endl
			<< "classical time step size = " << dtc << std::endl
			<< "# of classical steps = " << t_max/dtc << std::endl
			<< "velocity reversal mode: " << velo_rev << std::endl
			<< "velocity rescaling mode: " << velo_rescale << std::endl
			<< "friction mode: " << fric_mode << std::endl
			<< "temperature: " << kT << std::endl
			<< "omega = " << omega << std::endl
			<< "mass = " << mass << std::endl
			<< "x0_mpt = " << x0_mpt << std::endl
			<< "size of electronic basis: " << sz_elec << std::endl
			<< "size of electronic basis for FSSH: " << sz_elec_fssh << std::endl
			<< "has relaxation: " << has_rlx << std::endl
			<< "store n only: " << (bool)n_only << std::endl
			<< std::endl;
	}

	if (id != root) {
		set_size(sz_x, xgrid);
		set_size(sz_elec, sz_x, pes, n_imp, force, Gamma);
		set_size(sz_elec*sz_elec, sz_x, dc);
	}

	bcast(root, xgrid, pes, n_imp, force, Gamma, dc);

	////////////////////////////////////////////////////////////
	//					Model Setup
	////////////////////////////////////////////////////////////

	ModelInterp model(xgrid, pes.t(), n_imp.t(), force.t(), dc.t(), Gamma.t());

	if (id == nprocs-1) {
		std::cout << "Model Initialized" << std::endl;
	}

	uword ntc = t_max / dtc;
	vec time_grid;
	if (id == root) {
		time_grid = linspace(0, t_max, ntc);
		dtc = time_grid(1) - time_grid(0);
	}
	bcast(root, dtc);

	double fric_gamma;
	switch (fric_mode) {
		case -1:
			fric_gamma = 0.0;
			break;
		default:
			fric_gamma = 2.0 * mass * omega;
	}

	uword n_trajs_local = n_trajs / nprocs;
	int rem = n_trajs % nprocs;
	if (id < rem)
		n_trajs_local += 1;

	FSSH_rlx fssh_rlx( &model, mass, dtc, ntc, 
			kT, fric_gamma, velo_rev, velo_rescale, has_rlx, sz_elec_fssh, n_only);

	if (id == nprocs-1) {
		std::cout << "FSSH_rlx Initialized" << std::endl 
			<< "# of local trajectories (last proc) = " << n_trajs_local << std::endl;
	}

	// local data
	mat x_local, v_local, E_local, n_local;
	umat state_local, num_fhop_local;
	n_local.set_size(ntc, n_trajs_local);
	num_fhop_local.set_size(n_trajs_local);
	if (!n_only) {
		set_size({ntc, n_trajs_local}, x_local, v_local, state_local, E_local);
	}

	// global data
	mat x_t;
	mat n_t;
	mat v_t;
	mat E_t;
	umat state_t;
	umat num_fhop;

	if (id == root) {
		n_t.set_size(ntc, n_trajs);
		num_fhop.set_size(n_trajs);
		if (!n_only) {
			set_size({ntc, n_trajs}, x_t, v_t, state_t, E_t);
		}
		sw.run();
	}

	// Wigner quasi-probability distribution of harmonic oscillator
	double sigma_x, sigma_p;
	std::tie(sigma_x, sigma_p) = ho_wigner(mass, omega, kT);

	for (uword i = 0; i != n_trajs_local; ++i) {
		arma::arma_rng::set_seed_random();
		double x0 = x0_mpt + arma::randn()*sigma_x;
		double v0 = arma::randn() * sigma_p / mass;

		vec rho_eq = boltzmann(model.E(x0).head(sz_elec_fssh), kT);
		cx_mat rho0 = zeros<cx_mat>(sz_elec_fssh, sz_elec_fssh);
		rho0.diag() = conv_to<cx_vec>::from(rho_eq);

		uword state0 = 0;
		vec P_cumu = cumsum(rho_eq);
		double r = randu();
		for (state0 = 0; state0 != sz_elec_fssh; ++state0) {
			if (r < P_cumu(state0))	
				break;
		}


		fssh_rlx.initialize(state0, x0, v0, rho0);
		fssh_rlx.propagate();

		n_local.col(i) = fssh_rlx.n_t;
		num_fhop_local(i) = fssh_rlx.num_frustrated_hops;

		if (!n_only) {
			x_local.col(i) = fssh_rlx.x_t;
			v_local.col(i) = fssh_rlx.v_t;
			state_local.col(i) = fssh_rlx.state_t;
			E_local.col(i) = fssh_rlx.E_t;
		}

		if (nprocs == 1) {
			if (i == 0)
				std::cout << std::endl << std::endl;
			std::cout << "\033[A\033[2K\033[A\033[2K\r";
		}

		if (id == root)
			sw.report();

		std::cout << "proc id = " << id 
			<< "   local task: " << (i+1) << "/" << n_trajs_local << " finished"
			<< std::endl;
	}

	////////////////////////////////////////////////////////////
	//					collect data
	////////////////////////////////////////////////////////////
	MPI_Barrier(MPI_COMM_WORLD);
	if (id == root) {
		sw.report("start collecting data");
		arma_save<raw_binary>(savedir, time_grid, "t.dat");
	}

	MPI_Barrier(MPI_COMM_WORLD);
	gatherv(root, n_local, n_t);
	MPI_Barrier(MPI_COMM_WORLD);
	n_local.clear();
	if (id == root) {
		arma_save<raw_binary>(savedir, n_t, "n_t.dat");
		std::cout << "n saved" << std::endl;
		n_t.clear();
		std::cout << "n cleared" << std::endl;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	gatherv(root, num_fhop_local, num_fhop);
	MPI_Barrier(MPI_COMM_WORLD);
	num_fhop_local.clear();
	if (id == root) {
		arma_save<raw_binary>(savedir, num_fhop, "num_fhop.dat");
		std::cout << "fhop saved" << std::endl;
		num_fhop.clear();
		std::cout << "fhop cleared" << std::endl;
	}

	if (!n_only) {
		MPI_Barrier(MPI_COMM_WORLD);
		gatherv(root, x_local, x_t);
		MPI_Barrier(MPI_COMM_WORLD);
		x_local.clear();
		if (id == root) {
			arma_save<raw_binary>(savedir, x_t, "x_t.dat");
			std::cout << "x saved" << std::endl;
			x_t.clear();
			std::cout << "x cleared" << std::endl;
		}

		MPI_Barrier(MPI_COMM_WORLD);
		gatherv(root, state_local, state_t);
		MPI_Barrier(MPI_COMM_WORLD);
		state_local.clear();
		if (id == root) {
			mkdir(savedir);
			arma_save<raw_binary>(savedir, state_t, "state_t.dat");
			std::cout << "state saved" << std::endl;
			state_t.clear();
			std::cout << "state cleared" << std::endl;
		}

		MPI_Barrier(MPI_COMM_WORLD);
		gatherv(root, v_local, v_t);
		MPI_Barrier(MPI_COMM_WORLD);
		v_local.clear();
		if (id == root) {
			arma_save<raw_binary>(savedir, v_t, "v_t.dat");
			std::cout << "v saved" << std::endl;
			v_t.clear();
			std::cout << "v cleared" << std::endl;
		}

		MPI_Barrier(MPI_COMM_WORLD);
		gatherv(root, E_local, E_t);
		MPI_Barrier(MPI_COMM_WORLD);
		E_local.clear();
		if (id == root) {
			arma_save<raw_binary>(savedir, E_t, "E_t.dat");
			std::cout << "E saved" << std::endl;
			E_t.clear();
			std::cout << "E cleared" << std::endl;
		}
	}

	if (id == root) {
		sw.report("program end");
	}
	/*
	gatherv(root, state_local, state_t, x_local, x_t, v_local, v_t, E_local, E_t,
			num_fhop_local, num_fhop);

	if (id == root) {
		mkdir(savedir);
		arma_save<raw_binary>( savedir,
				state_t, "state_t.dat",
				x_t, "x_t.dat",
				v_t, "v_t.dat",
				E_t, "E_t.dat",
				num_fhop, "num_fhop.dat",
				time_grid, "t.dat"
		);
		sw.report("program end");
	}
	*/

	MPI_Finalize();

	return 0;
}

