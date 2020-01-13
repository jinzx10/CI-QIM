#include <TwoPara.h>
#include <chrono>
#include <mpi.h>
#include <cstdlib>

using namespace arma;
using iclock = std::chrono::high_resolution_clock;

int main() {

	int id, nprocs;
	int num_trajs = 960;
	std::string datadir = "/home/zuxin/job/cme/data/20191114/";
	std::string cmd;
	const char* system_cmd = nullptr;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	double x0_mpt = 2;
	double x0_fil = 2.3;
	double omega= 0.002;
	double mass = 14000;
	double dE_fil = 0.0000;
	
	auto E_mpt = [&](double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_mpt) * (x - x0_mpt);};
	auto E_fil = [&](double const& x) { return 0.5 * mass * omega* omega* 
		(x - x0_fil) * (x - x0_fil) + dE_fil;};

	double W = 0.05;
	double bath_min = -W;
	double bath_max = W;
	uword n_bath = 800;
	vec bath = linspace<vec>(bath_min, bath_max, n_bath);
	double dos = 1.0 / (bath(1) - bath(0));

	double Gamma = 0.001;
	auto cpl = [&](double const& x) -> vec {
		return ones<vec>(n_bath) * sqrt(Gamma/2/datum::pi/dos);
	};

	uword n_occ = n_bath / 2;

	uword nx = 100;
	vec xgrid = linspace(x0_mpt-0.5, x0_fil+0.5, nx);

	iclock::time_point start;
	std::chrono::duration<double> dur;

	TwoPara model(E_mpt, E_fil, bath, cpl, n_occ);

	start = iclock::now();
	model.calc(xgrid(0));
	dur = iclock::now() - start;

	std::cout << "time elapsed = " << dur.count() << std::endl;

	model.Gamma.save("test_TwoPara.txt", arma::raw_ascii);
	
	return 0;
}
