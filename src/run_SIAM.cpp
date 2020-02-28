#include <mpi.h>
#include <string>
#include "mpi_helper.h"
#include "widgets.h"
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




	MPI_Finalize();

	return 0;
}
