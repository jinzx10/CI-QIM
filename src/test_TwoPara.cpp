#include <TwoPara.h>
#include <chrono>
#include <mpi.h>
#include <cstdlib>

using namespace arma;
using iclock = std::chrono::high_resolution_clock;

int main() {

	int id, nprocs;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	iclock::time_point start;
	std::chrono::duration<double> dur;
	std::string datadir = "/home/zuxin/job/CI-QIM/data/test_TwoPara/Gamma/00016_800/";
	std::string cmd;
	const char* system_cmd = nullptr;
	int status;

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

	double W = 0.1;
	double bath_min = -W;
	double bath_max = W;
	uword n_bath = 800;
	vec bath = linspace<vec>(bath_min, bath_max, n_bath);
	double dos = 1.0 / (bath(1) - bath(0));

	uword n_occ = n_bath / 2;
	uword n_vir = n_bath + 1 - n_occ;
	uword sz_sub = n_occ + n_vir - 1;

	double hybrid = 0.0016;
	vec cpl = ones<vec>(n_bath) * sqrt(hybrid/2/datum::pi/dos);

	uword nx = 96*3;
	vec xgrid = linspace(-10, 30, nx);
	int local_nx = nx / nprocs;

	// global
	mat Gamma, val_cis_sub, force, dc;
	vec Eg, V0, n_imp;

	// local
	mat local_Gamma = zeros(sz_sub, local_nx);
	mat local_val_cis_sub = zeros(sz_sub, local_nx);
	mat local_force = zeros(sz_sub+1, local_nx);
	vec local_Eg = zeros(local_nx);
	vec local_n_imp = zeros(local_nx);
	vec local_V0 = zeros(local_nx);

	uword sz_dc = 2;
	mat local_dc = zeros(sz_dc*sz_dc, local_nx);

	if (id == 0) {
		Gamma.zeros(sz_sub, nx);
		val_cis_sub.zeros(sz_sub, nx);
		force.zeros(sz_sub+1, nx);
		dc.zeros(sz_dc*sz_dc, nx);
		Eg.zeros(nx);
		V0.zeros(nx);
		n_imp.zeros(nx);

		start = iclock::now();
	}

	TwoPara model(E_mpt, E_fil, bath, cpl, n_occ);

	for (int i = 0; i != local_nx; ++i) {
		double x = xgrid(id*local_nx+i);
		model.set_and_calc(x);

		local_val_cis_sub.col(i) = model.val_cis_sub;
		local_Gamma.col(i) = model.Gamma;
		local_V0(i) = E_mpt(x);
		local_Eg(i) = model.ev_H;
		local_n_imp(i) = model.ev_n;
		local_force.col(i) = model.force();
		local_dc.col(i) = model.dc(sz_dc).as_col();

		std::cout << id*local_nx+i+1 << "/" << nx << " finished" << std::endl;
	}

	::MPI_Gather(local_val_cis_sub.memptr(), local_val_cis_sub.n_elem, MPI_DOUBLE, val_cis_sub.memptr(), local_val_cis_sub.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	::MPI_Gather(local_force.memptr(), local_force.n_elem, MPI_DOUBLE, force.memptr(), local_force.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	::MPI_Gather(local_Gamma.memptr(), local_Gamma.n_elem, MPI_DOUBLE, Gamma.memptr(), local_Gamma.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	::MPI_Gather(local_dc.memptr(), local_dc.n_elem, MPI_DOUBLE, dc.memptr(), local_dc.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	::MPI_Gather(local_V0.memptr(), local_V0.n_elem, MPI_DOUBLE, V0.memptr(), local_V0.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	::MPI_Gather(local_Eg.memptr(), local_Eg.n_elem, MPI_DOUBLE, Eg.memptr(), local_Eg.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	::MPI_Gather(local_n_imp.memptr(), local_n_imp.n_elem, MPI_DOUBLE, n_imp.memptr(), local_n_imp.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	::MPI_Barrier(MPI_COMM_WORLD);

	if (id == 0) {
#ifndef NO_WRITE
		cmd = "mkdir -p " + datadir;
		system_cmd = cmd.c_str();
		status = std::system(system_cmd);

		xgrid.save(datadir+"xgrid.txt", arma::raw_ascii);
		Gamma.save(datadir+"Gamma.txt", arma::raw_ascii);
		val_cis_sub.save(datadir+"val_cis_sub.txt", arma::raw_ascii);
		force.save(datadir+"force.txt", arma::raw_ascii);
		dc.save(datadir+"dc.txt", arma::raw_ascii);
		V0.save(datadir+"V0.txt", arma::raw_ascii);
		Eg.save(datadir+"Eg.txt", arma::raw_ascii);
		n_imp.save(datadir+"n_imp.txt", arma::raw_ascii);
#endif

		dur = iclock::now() - start;
		std::cout << "time elapsed = " << dur.count() << std::endl;
	}

	::MPI_Barrier(MPI_COMM_WORLD);
	::MPI_Finalize();
	
	return 0;
}
