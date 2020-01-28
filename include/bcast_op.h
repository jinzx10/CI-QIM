#ifndef __BROADCASTED_OPERATIONS_H__
#define __BROADCASTED_OPERATIONS_H__

#include <armadillo>

template <typename Op>
arma::mat bcast_op(arma::vec const& v, arma::rowvec const& r, Op op) {
	arma::mat result(v.n_rows, r.n_cols);
	for (arma::uword j = 0; j != r.n_cols; ++j) {
		result.col(j) = op(v, r(j));
	}
	return result;
}

#endif
