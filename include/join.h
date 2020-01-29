#ifndef __ARMA_JOIN_H__
#define __ARMA_JOIN_H__

#include <armadillo>
#include <initializer_list>

template <typename T>
T join_r(std::initializer_list<T> m) {
	T z;
	for (auto it = m.begin(); it != m.end(); ++it) {
		z = join_rows(z, *it);
	}
	return z;
}

template <typename T>
T join(std::initializer_list< std::initializer_list<T> > m) {
	T z;
	for (auto it = m.begin(); it != m.end(); ++it) {
		z = join_cols(z, join_r(*it));
	}
	return z;
}

inline arma::mat join_d(arma::mat const& m) {
	return m;
}

inline arma::mat join_d(arma::mat const& m1, arma::mat const& m2) {
    return join_cols(
            join_rows(m1, arma::zeros(m1.n_rows, m2.n_cols)),
            join_rows(arma::zeros(m2.n_rows, m1.n_cols), m2) );
}

template <typename ...Ts>
arma::mat join_d(arma::mat const& m, Ts const& ...args) {
    return join_d(m, join_d(args...));
}


inline arma::uvec cat() {
	return arma::uvec{}; 
}

template <typename T>
arma::uvec cat(T const& i) {
	return arma::uvec{i};
}

template <typename T, typename ...Ts>
arma::uvec cat(T const& i, Ts const& ...args) {
	return arma::join_cols(arma::uvec{i}, cat(args...));
}

#endif
