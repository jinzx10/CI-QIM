#include <mpi.h>
#include <chrono>
#include <functional>
#include <armadillo>
#include "arma_mpi_helper.h"
#include "auxmath.h"

using namespace arma;
using iclock = std::chrono::high_resolution_clock;

struct TwoPara
{
	using d2d = std::function<double(double)>;

	TwoPara( d2d E_mpt_, d2d E_fil_ );

	d2d E0;
	d2d F0;

	d2d E_mpt;
	d2d E_fil;
	d2d dE_mpt;
	d2d dE_fil;
};

TwoPara::TwoPara( d2d E_mpt_, d2d E_fil_ ):
	E_mpt(E_mpt_), E_fil(E_fil_)
{
	dE_mpt = grad(E_mpt);
	dE_fil = grad(E_fil);

	E0 = [this] (double const& x) {
		return x < 8.00011 ? E_mpt(x) : E_fil(x);
	};

	F0 = [this] (double const& x) {
		return x < 8.00011 ? -dE_mpt(x) : -dE_fil(x);
	};
}


struct Langevin
{
	Langevin(
			TwoPara*					model_,
			double			const&		mass_,
			double			const&		dtc_,
			arma::uword		const& 		ntc_,
			double			const& 		kT_,
			double 			const& 		gamma_
	);

	void						initialize(double const& x0_, double const& v0_);
	void						propagate();

	void						evolve_nucl(); // Velocity Verlet
	void 						collect();
	void						clear();

	double						energy();

	TwoPara*		const		model;
	double			const		mass;
	double 			const		dtc;
	arma::uword		const 		ntc; // number of classical time steps
	double			const		kT;
	double 			const		gamma; // external phononic friction

	double						x;
	double 						v;

	arma::uword					counter;

	// data storage for one trajectory
	arma::vec					x_t;
	arma::vec					v_t;
	arma::vec					E_t;
};


Langevin::Langevin( 
		TwoPara*					model_,
		double			const&		mass_,
		double			const&		dtc_,
		uword			const& 		ntc_,
		double			const&		kT_,
		double			const&		gamma_
):
	model(model_), mass(mass_), dtc(dtc_), ntc(ntc_),
	kT(kT_), gamma(gamma_), x(0), v(0), counter(0),
	x_t(zeros(ntc)), v_t(zeros(ntc)), E_t(zeros(ntc))
{}

void Langevin::initialize(double const& x0, double const& v0) {
	clear();
	x = x0;
	v = v0;
	collect();
}

void Langevin::evolve_nucl() {
	// Velocity-Verlet (with external phononic friction)
	double F_fric = -gamma * v;
	double F_rand = sqrt( 2.0 * gamma * kT / dtc ) * randn();
	double F_pes = model->F0(x);
	double a = ( F_pes + F_fric + F_rand ) / mass;
	x += v * dtc + 0.5 * a * dtc * dtc;
	F_pes = model->F0(x);
	double a_new = ( F_pes + F_fric + F_rand ) / mass;
	v += 0.5 * (a + a_new) * dtc;
}


double Langevin::energy() {
	double E_kin = 0.5 * mass * v * v;
	double E_elec = model->E0(x);
	return E_kin + E_elec;
}

void Langevin::collect() {
	x_t(counter) = x;
	v_t(counter) = v;
	E_t(counter) = energy();
}

void Langevin::clear() {
	counter = 0;
	x_t.zeros();
	v_t.zeros();
	E_t.zeros();
}

void Langevin::propagate() {
	for (counter = 1; counter != ntc; ++counter) {
		evolve_nucl();
		collect();
	}
}


int main() {

	int id, nprocs;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	iclock::time_point start;
	std::chrono::duration<double> dur;
	std::string savedir = "/home/zuxin/job/CI-QIM/data/Langevin_ref/";
	std::string command = "mkdir -p " + savedir;

	////////////////////////////////////////////////////////////
	//					Two-Parabola model
	////////////////////////////////////////////////////////////
	double x0_mpt = 0;
	double x0_fil = 20.6097;
	double omega = 0.0002;
	double mass = 2000;
	double dE_fil = -0.0038;
	
	auto E_mpt = [&] (double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_mpt) * (x - x0_mpt);};
	auto E_fil = [&] (double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_fil) * (x - x0_fil) + dE_fil;};

	TwoPara model(E_mpt, E_fil);


	////////////////////////////////////////////////////////////
	//			Born-Oppenheimer Molecular Dynamics
	////////////////////////////////////////////////////////////
	double t_max = 1e6;
	double dtc = 2;
	int n_trajs = 1920;
	uword ntc = t_max / dtc;
	vec time_grid;
	if (id == 0) {
		time_grid = linspace(0, t_max, ntc);
		dtc = time_grid(1) - time_grid(0);
	}
	bcast(&dtc);

	double fric_gamma = 2.0 * mass * omega;
	double kT = 9.5e-4;

	int n_trajs_local = n_trajs / nprocs;
	Langevin lgv(&model, mass, dtc, ntc, kT, fric_gamma);

	// local data
	mat x_local, v_local, E_local;

	set_size(ntc, n_trajs_local, x_local, v_local, E_local);

	// global data
	mat x_t, v_t, E_t;

	if (id == 0) {
		set_size(ntc, n_trajs, x_t, v_t, E_t);
		start = iclock::now();
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

		lgv.initialize(x0, v0);
		lgv.propagate();

		x_local.col(i) = lgv.x_t;
		v_local.col(i) = lgv.v_t;
		E_local.col(i) = lgv.E_t;
	}

	gather( x_local, x_t, v_local, v_t, E_local, E_t );

	////////////////////////////////////////////////////////////
	//					save data
	////////////////////////////////////////////////////////////
	if (id == 0) {
		bool status = std::system(command.c_str());
		if (status) {
			savedir = "./";
			command = "mkdir -p " + savedir;
			status = std::system(command.c_str());
		}

		arma_save<raw_binary>( savedir, 
				x_t, "x.dat",
				v_t, "v.dat",
				E_t, "E.dat",
				time_grid, "t.dat"
		);

		dur = iclock::now() - start;
		std::cout << dur.count() << std::endl;
	}

	::MPI_Finalize();


}
