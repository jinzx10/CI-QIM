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


template <typename T1, typename T2, typename Op>
using return_t = decltype( std::declval<Op>()( std::declval<T1>(), std::declval<T2>() ) );

template <typename T1, typename T2, typename Op, typename = void>
struct op_is_defined : std::false_type {};

template <typename ...>
using void_t = void;

template <typename T1, typename T2, typename Op>
struct op_is_defined<T1, T2, Op, void_t<decltype( std::declval<Op>()( std::declval<T1>(), std::declval<T2>() ) )>> : std::true_type 
{
	using type = decltype( std::declval<Op>()( std::declval<T1>(), std::declval<T2>() ) );
};


template <typename C, typename R, typename Op>
typename std::enable_if< C::is_col && R::is_row, arma::Mat< typename return_t<C, typename R::elem_type, Op>::elem_type > >::type bcast_test(C const& col, R const& row, Op op) {
	arma::Mat< typename return_t<C, typename R::elem_type, Op>::elem_type > result( arma::size(col).n_rows, arma::size(row).n_cols);
	for (arma::uword j = 0; j != arma::size(row).n_cols; ++j) {
		result.col(j) = op(col, arma::conv_to<arma::Row<typename R::elem_type>>::from(row)(j));
	}
	return result;
}

template <typename R, typename C, typename Op>
typename std::enable_if< R::is_row && C::is_col, arma::Mat< typename return_t<typename R::elem_type, C, Op>::elem_type > >::type bcast_test(R const& row, C const& col, Op op) {
	arma::Mat< typename return_t<typename R::elem_type, C, Op>::elem_type > result( arma::size(col).n_rows, arma::size(row).n_cols);
	for (arma::uword j = 0; j != arma::size(row).n_cols; ++j) {
		result.col(j) = op(arma::conv_to<arma::Row<typename R::elem_type>>::from(row)(j), col);
	}
	return result;
}

template <typename C, typename R, typename Op>
typename std::enable_if< C::is_col && R::is_row && !op_is_defined<C, typename R::elem_type, Op>::value, arma::Mat< typename return_t<typename C::elem_type, R, Op>::elem_type > >::type bcast_test(C const& col, R const& row, Op op) {
	arma::Mat< typename return_t<typename C::elem_type, R, Op>::elem_type > result( arma::size(col).n_rows, arma::size(row).n_cols);
	for (arma::uword i = 0; i != arma::size(col).n_rows; ++i) {
		result.row(i) = op(arma::conv_to<arma::Col<typename C::elem_type>>::from(col)(i), row);
	}
	return result;
}


int main() {

	vec x = {-3, -1, 0, 0.5, 0.8, 0.9, 1};
	vec y = exp(-x%x/2);
	rowvec xr = x.t();
	rowvec yr = y.t();

	std::cout << op_is_defined<decltype(x), typename decltype(y)::elem_type, std::plus<>>::value << std::endl;

	mat z1 = bcast_test(x, y.t(), std::plus<>());

	std::cout << x.is_vec()<< std::endl;
	std::cout << x.t().is_vec()<< std::endl;
	std::cout << xr.is_vec()<< std::endl;
	std::cout << xr.t().is_vec()<< std::endl;

	
	return 0;
}
