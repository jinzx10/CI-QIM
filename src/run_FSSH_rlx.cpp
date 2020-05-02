#include <mpi.h>
#include "FSSH_rlx.h"
#include "ModelInterp.h"
#include "mpi_helper.h"
#include "arma_helper.h"
#include "widgets.h"

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
	std::string readdir;
	std::string savedir;
	int n_trajs;
	double t_max;
	double dtc;
	
	vec xgrid;
	mat pes, force, Gamma, dc;
	uword sz_x, sz_elec;

	if (id == 0) {
		readargs(argv, readdir, savedir, n_trajs, t_max, dtc);
		std::cout << "data will be read from: " << readdir << std::endl;
		std::cout << "data will be saved to: " << savedir << std::endl;
		std::cout << "# of trajectories = " << n_trajs << std::endl;
		std::cout << "maximun time = " << t_max << std::endl;
		std::cout << "classical time step size = " << dtc << std::endl;
		std::cout << "# of classical steps = " << t_max/dtc << std::endl;

		arma_load( readdir, 
				xgrid, "xgrid.dat",
				pes, "E.dat",
				force, "F.dat",
				Gamma, "Gamma.dat",
				dc, "dc.dat"
		);

		sz_x = xgrid.n_elem;
		sz_elec = pes.n_elem / sz_x;

		pes.reshape(sz_elec, sz_x);
		force.reshape(sz_elec, sz_x);
		Gamma.reshape(sz_elec, sz_x);
		dc.reshape(sz_elec*sz_elec, sz_x);

		std::cout << "data read successfully" << std::endl;
		std::cout << "# of coarse xgrid points: " << sz_x << std::endl;
		std::cout << "size of electronic basis: " << sz_elec << std::endl;
	}

	bcast(readdir, n_trajs, t_max, dtc, sz_x, sz_elec);

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
	double omega = 0.0002;
	double mass = 2000;
	double x0_mpt = 0;

	uword ntc = t_max / dtc;
	vec time_grid;
	if (id == 0) {
		time_grid = linspace(0, t_max, ntc);
		dtc = time_grid(1) - time_grid(0);
	}
	bcast(dtc);

#ifdef NO_FRICTION
	double fric_gamma = 0;
#else
	double fric_gamma = 2.0 * mass * omega;
#endif

	double kT = 9.5e-4;
	int n_trajs_local = n_trajs / nprocs;
	int rem = n_trajs % nprocs;
	if (id < rem)
		n_trajs_local += 1;

	arma::uword sz_rho = 2;
	FSSH fssh(&model, mass, dtc, ntc, kT, fric_gamma);

	// local data
	mat x_local, v_local, E_local;
	umat state_local;
	set_size(ntc, n_trajs_local, x_local, v_local, state_local, E_local);

	// global data
	mat x_t;
	mat v_t;
	umat state_t;
	mat E_t;

	if (id == 0) {
		set_size(ntc, n_trajs, x_t, v_t, state_t, E_t);
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

		vec val = model.E_adi(x0);
		vec rho_eq = exp(-(val-val(0))/kT) / accu( exp(-(val-val(0))/kT) );
		cx_mat rho0 = zeros<cx_mat>(sz_rho, sz_rho);
		rho0(0,0) = rho_eq(0);
		rho0(1,1) = rho_eq(1);
		uword state0 = (arma::randu() < rho_eq(0)) ? 0 : 1;

		fssh.initialize(state0, x0, v0, rho0);
		fssh.propagate();

		x_local.col(i) = fssh.x_t;
		v_local.col(i) = fssh.v_t;
		state_local.col(i) = fssh.state_t;
		E_local.col(i) = fssh.E_t;
	}

	gatherv(state_local, state_t, x_local, x_t, v_local, v_t, E_local, E_t);

	////////////////////////////////////////////////////////////
	//					save data
	////////////////////////////////////////////////////////////
	if (id == 0) {
		mkdir(savedir);
		arma_save<raw_binary>( savedir,
				state_t, "state.dat",
				x_t, "x.dat",
				v_t, "v.dat",
				E_t, "E.dat",
				time_grid, "t.dat"
		);
		sw.report();
	}

	MPI_Finalize();

	return 0;
}

