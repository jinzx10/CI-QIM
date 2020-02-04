#ifndef __ARMA_MPI_HELPER_H__
#define __ARMA_MPI_HELPER_H__

#include <armadillo>
#include <mpi.h>

inline void gather() {
}

inline void gather(arma::vec const& local, arma::mat& global) {
	::MPI_Gather(local.memptr(), local.n_elem, MPI_DOUBLE, global.memptr(), local.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

inline void gather(arma::uvec const& local, arma::umat& global) {
	::MPI_Gather(local.memptr(), local.n_elem, MPI_UNSIGNED_LONG_LONG, global.memptr(), local.n_elem, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
}

inline void gather(arma::vec const& local, arma::vec& global) {
	::MPI_Gather(local.memptr(), local.n_elem, MPI_DOUBLE, global.memptr(), local.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

inline void gather(arma::uvec const& local, arma::uvec& global) {
	::MPI_Gather(local.memptr(), local.n_elem, MPI_UNSIGNED_LONG_LONG, global.memptr(), local.n_elem, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
}

template <typename eT, typename ...Ts>
void gather(arma::Col<eT> const& local, arma::Mat<eT>& global, Ts& ...args) {
	gather(local, global);
	gather(args...);
}

template <typename eT, typename ...Ts>
void gather(arma::Col<eT> const& local, arma::Col<eT>& global, Ts& ...args) {
	gather(local, global);
	gather(args...);
}

#endif
