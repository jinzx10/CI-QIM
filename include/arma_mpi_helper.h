#ifndef __ARMA_MPI_HELPER_H__
#define __ARMA_MPI_HELPER_H__

#include <armadillo>
#include <mpi.h>

template <typename eT>
MPI_Datatype mpi_type_helper() {
	std::cerr << "fails to convert to MPI datatype" << std::endl;
	return nullptr;
}

template<>
inline MPI_Datatype mpi_type_helper<char>() {
	return MPI_CHAR;
}

template<>
inline MPI_Datatype mpi_type_helper<unsigned char>() {
	return MPI_UNSIGNED_CHAR;
}

template<>
inline MPI_Datatype mpi_type_helper<float>() {
	return MPI_FLOAT;
}

template<>
inline MPI_Datatype mpi_type_helper<double>() {
	return MPI_DOUBLE;
}

template<>
inline MPI_Datatype mpi_type_helper<long double>() {
	return MPI_LONG_DOUBLE;
}

template<>
inline MPI_Datatype mpi_type_helper<short>() {
	return MPI_SHORT;
}

template<>
inline MPI_Datatype mpi_type_helper<int>() {
	return MPI_INT;
}

template<>
inline MPI_Datatype mpi_type_helper<long>() {
	return MPI_LONG;
}

template<>
inline MPI_Datatype mpi_type_helper<long long>() {
	return MPI_LONG_LONG;
}

template<>
inline MPI_Datatype mpi_type_helper<unsigned short>() {
	return MPI_UNSIGNED_SHORT;
}

template<>
inline MPI_Datatype mpi_type_helper<unsigned int>() {
	return MPI_UNSIGNED;
}

template<>
inline MPI_Datatype mpi_type_helper<unsigned long>() {
	return MPI_UNSIGNED_LONG;
}

template<>
inline MPI_Datatype mpi_type_helper<unsigned long long>() {
	return MPI_UNSIGNED_LONG_LONG;
}


template <typename eT>
void gather(eT* const& ptr_local, arma::Mat<eT>& global, int root = 0) {
	MPI_Gather(ptr_local, 1, mpi_type_helper<eT>(), global.memptr(), 1, mpi_type_helper<eT>(), root, MPI_COMM_WORLD);
}

template <typename eT>
void gather(arma::Mat<eT> const& local, arma::Mat<eT>& global, int root = 0) {
	MPI_Gather(local.memptr(), local.n_elem, mpi_type_helper<eT>(), global.memptr(), local.n_elem, mpi_type_helper<eT>(), root, MPI_COMM_WORLD);
}

template <typename eT, typename ...Ts>
void gather(eT* const& ptr_local, arma::Mat<eT>& global, Ts& ...args) {
	gather(ptr_local, global);
	gather(args...);
}

template <typename eT, typename ...Ts>
void gather(arma::Mat<eT> const& local, arma::Mat<eT>& global, Ts& ...args) {
	gather(local, global);
	gather(args...);
}



template <typename eT>
void bcast(eT* ptr, int root = 0) {
	MPI_Bcast(ptr, 1, mpi_type_helper<eT>(), root, MPI_COMM_WORLD);
}

template <typename eT>
void bcast(arma::Mat<eT>& data, int root = 0) {
	MPI_Bcast(data.memptr(), data.n_elem, mpi_type_helper<eT>(), root, MPI_COMM_WORLD);
}

template <typename eT, typename ...Ts>
void bcast(eT* ptr, Ts& ...args) {
	bcast(ptr);
	bcast(args...);
}

template <typename eT, typename ...Ts>
void bcast(arma::Mat<eT>& data, Ts& ...args) {
	bcast(data);
	bcast(args...);
}



#endif
