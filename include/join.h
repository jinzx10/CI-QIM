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

inline arma::uvec cat() { return arma::uvec{}; }

template <typename T>
arma::uvec cat(T const& i) {
	return arma::uvec{i};
}

template <typename T, typename ...Ts>
arma::uvec cat(T const& i, Ts const& ...args) {
	return arma::join_cols(arma::uvec{i}, cat(args...));
}

#endif
