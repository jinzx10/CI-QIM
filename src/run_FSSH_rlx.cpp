#include <mpi.h>
#include "FSSH_rlx.h"
#include "ModelInterp.h"
#include "mpi_helper.h"
#include "arma_helper.h"
#include "widgets.h"
#include "math_helper.h"

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
	std::string input_file;
	Parser p({"readdir", "savedir", "n_trajs", "t_max", "dtc", 
			"velo_rev", "fric_mode", "kT"});

	std::string readdir;
	std::string savedir;
	int n_trajs;
	double t_max;
	double dtc;
	int velo_rev;
	int fric_mode;
	double kT;

	double omega;
	double mass;
	double x0_mpt;
	
	vec xgrid;
	mat pes, force, Gamma, dc;
	uword sz_x, sz_elec;

	std::string paramfile;
	if (id == 0) {
		readargs(argv, input_file);

		p.parse(input_file);
		p.pour(readdir, savedir, n_trajs, t_max, dtc, velo_rev, fric_mode, kT);

		paramfile = readdir + "/param.txt";
		p.reset({"omega", "mass", "x0_mpt"});
		p.parse(paramfile);
		p.pour(omega, mass, x0_mpt);

		arma_load( readdir, 
				xgrid, "xgrid.dat",
				pes, "E_adi.dat",
				force, "F_adi.dat",
				Gamma, "Gamma_rlx.dat",
				dc, "dc_adi.dat"
		);

		sz_x = xgrid.n_elem;
		sz_elec = pes.n_elem / sz_x;

		pes.reshape(sz_elec, sz_x);
		force.reshape(sz_elec, sz_x);
		Gamma.reshape(sz_elec, sz_x);
		dc.reshape(sz_elec*sz_elec, sz_x);

		std::cout << "data are read from: " << readdir << std::endl
			<< "data will be saved to: " << savedir << std::endl
			<< "# of trajectories = " << n_trajs << std::endl
			<< "maximun time = " << t_max << std::endl
			<< "classical time step size = " << dtc << std::endl
			<< "# of classical steps = " << t_max/dtc << std::endl
			<< "velocity reversal mode: " << velo_rev << std::endl
			<< "friction mode: " << fric_mode << std::endl
			<< "temperature: " << kT << std::endl
			<< "omega = " << omega << std::endl
			<< "mass = " << mass << std::endl
			<< "x0_mpt = " << x0_mpt << std::endl
			<< "size of electronic basis: " << sz_elec << std::endl
			<< std::endl;
	}

	bcast(n_trajs, t_max, dtc, velo_rev, fric_mode, kT, 
			omega, mass, x0_mpt, sz_x, sz_elec);

	////////////////////////////////////////////////////////////
	//					Model Setup
	////////////////////////////////////////////////////////////

	if (id != 0) {
		set_size(sz_x, xgrid);
		set_size(sz_elec, sz_x, pes, force, Gamma);
		set_size(sz_elec*sz_elec, sz_x, dc);
	}

	bcast(xgrid, pes, force, Gamma, dc);

	ModelInterp model(xgrid, pes.t(), force.t(), dc.t(), Gamma.t());

	////////////////////////////////////////////////////////////
	//			Fewest-Switches Surface Hopping
	////////////////////////////////////////////////////////////
	uword ntc = t_max / dtc;
	vec time_grid;
	if (id == 0) {
		time_grid = linspace(0, t_max, ntc);
		dtc = time_grid(1) - time_grid(0);
	}
	bcast(dtc);

	double fric_gamma;
	switch (fric_mode) {
		case -1:
			fric_gamma = 0.0;
			break;
		default:
			fric_gamma = 2.0 * mass * omega;
	}

	int n_trajs_local = n_trajs / nprocs;
	int rem = n_trajs % nprocs;
	if (id < rem)
		n_trajs_local += 1;

	FSSH_rlx fssh_rlx(&model, mass, dtc, ntc, kT, fric_gamma, velo_rev);

	// local data
	mat x_local, v_local, E_local;
	umat state_local, num_fhop_local;
	set_size(ntc, n_trajs_local, x_local, v_local, state_local, E_local);
	set_size(n_trajs_local, num_fhop_local);

	// global data
	mat x_t;
	mat v_t;
	mat E_t;
	umat state_t;
	umat num_fhop;

	if (id == 0) {
		set_size(ntc, n_trajs, x_t, v_t, state_t, E_t);
		set_size(n_trajs, num_fhop);
		sw.run();
	}

	// Wigner quasi-probability distribution of harmonic oscillator
	// ground state
	//double sigma_x = std::sqrt(0.5/mass/omega);
	//double sigma_v = std::sqrt(omega/mass/2.0);
	
	// thermally-averaged
	double sigma_x = std::sqrt( 0.5 / mass / omega / std::tanh(omega/2.0/kT) );
	double sigma_p = std::sqrt( mass * omega / 2.0 / std::tanh(omega/2.0/kT) );
	arma::arma_rng::set_seed_random();

	for (int i = 0; i != n_trajs_local; ++i) {
		double x0 = x0_mpt + arma::randn()*sigma_x;
		double v0 = arma::randn() * sigma_p / mass;

		vec rho_eq = boltzmann(model.E(x0), kT);
		cx_mat rho0 = zeros<cx_mat>(sz_elec, sz_elec);
		rho0.diag() = conv_to<cx_vec>::from(rho_eq);

		uword state0 = 0;
		vec P_cumu = cumsum(rho_eq);
		double r = randu();
		for (state0 = 0; state0 != sz_elec; ++state0) {
			if (r < P_cumu(state0))	
				break;
		}

		fssh_rlx.initialize(state0, x0, v0, rho0);
		fssh_rlx.propagate();

		x_local.col(i) = fssh_rlx.x_t;
		v_local.col(i) = fssh_rlx.v_t;
		state_local.col(i) = fssh_rlx.state_t;
		E_local.col(i) = fssh_rlx.E_t;
		num_fhop_local(i) = fssh_rlx.num_frustrated_hops;

		if (nprocs == 1) {
			if (i == 0)
				std::cout << std::endl << std::endl;
			std::cout << "\033[A\033[2K\033[A\033[2K\r";
		}

		if (id == 0)
			sw.report();

		std::cout << "proc id = " << id 
			<< "   local task: " << (i+1) << "/" << n_trajs_local << " finished"
			<< std::endl;
	}

	gatherv(state_local, state_t, x_local, x_t, v_local, v_t, E_local, E_t,
			num_fhop_local, num_fhop);

	////////////////////////////////////////////////////////////
	//					save data
	////////////////////////////////////////////////////////////
	if (id == 0) {
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

	MPI_Finalize();

	return 0;
}

