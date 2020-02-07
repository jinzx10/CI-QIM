#include <iostream>
#include <armadillo>
#include <type_traits>

using namespace arma;

template <typename eT, typename Op>
arma::Mat<eT> bcast_op(arma::Col<eT> const& v, arma::Row<eT> const& r, Op op) {
	arma::Mat<eT> result(v.n_rows, r.n_cols);
	for (arma::uword j = 0; j != r.n_cols; ++j) {
		result.col(j) = op(v, r(j));
	}
	return result;
}

template <typename eT, typename Op>
arma::Mat<eT> bcast_op(arma::Row<eT> const& r, arma::Col<eT> const& v, Op op) {
	arma::Mat<eT> result(v.n_rows, r.n_cols);
	for (arma::uword j = 0; j != r.n_cols; ++j) {
		result.col(j) = op(r(j), v);
	}
	return result;
}

struct foo
{
	static const bool key = 1;
};

struct bar
{
	static const bool key = 0;
};

template <typename T>
typename std::enable_if<T::key, double>::type test() {
	std::cout << "double used" << std::endl;
}

template <typename T>
typename std::enable_if<!T::key, char>::type test() {
	std::cout << "char used" << std::endl;
}

template <typename T1, typename T2, typename Op>
typename std::enable_if<T1::is_col && T2::is_row, arma::Mat<typename T1::elem_type> >::type bcast(T1 const& v, T2 const& r, Op op) {
	typename std::enable_if<T1::is_col && T2::is_row, arma::Mat<typename T1::elem_type> >::type result(v.n_rows, r.n_cols);
	for (arma::uword j = 0; j != r.n_cols; ++j) {
		result.col(j) = op(v, r(j));
	}
	return result;
}

int main() {

	vec x = {-3, -1, 0, 0.5, 0.8, 0.9, 1};
	vec y = exp(-x%x/2);

	mat z = bcast(x,y.t(), std::plus<>());
	z.print();

	typedef Op<Col<double>, op_htrans> fr;

	std::cout << arma::Col<double>::is_col << std::endl;
	std::cout << Op<Col<double>, op_htrans>::is_col << std::endl;
	std::cout << Op<Col<double>, op_htrans>::is_row << std::endl;
	std::cout << arma::is_Col<vec>::value << std::endl;

	test<foo>();
	test<bar>();

	return 0;
}
