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
struct return_type { using type = decltype( std::declval<Op>()(  std::declval<T1>(), std::declval<T2>() ) ); };

template <typename T1, typename T2, typename Op>
using return_type_t = typename return_type<T1, T2, Op>::type;

template <typename T1, typename T2, typename Op>
typename std::enable_if< T1::is_col && T2::is_row, arma::Mat< return_type_t<typename T1::elem_type, typename T2::elem_type, Op> > >::type bcast(T1 const& v, T2 const& r, Op op) {
	typename std::enable_if<T1::is_col && T2::is_row, arma::Mat< return_type_t< typename T1::elem_type, typename T2::elem_type, Op>  > >::type result( arma::size(v).n_rows, arma::size(r).n_cols);
	for (arma::uword j = 0; j != arma::size(r).n_cols; ++j) {
		result.col(j) = op(v, arma::conv_to<arma::Row<typename T1::elem_type>>::from(r)(j));
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

	decltype( std::plus<>()(1,3.3) ) a;
	decltype( std::declval<std::plus<>>().operator()(1,3.3) ) b;

	std::result_of<std::plus<>(char,double)>::type c;

	std::cout << typeid( std::plus<>()(1,3.3) ).name() << std::endl;
	std::cout << typeid( a ).name() << std::endl;
	std::cout << typeid( b ).name() << std::endl;
	std::cout << typeid( c ).name() << std::endl;

	a = std::plus<>()(1,3.3);

	std::cout << typeid( return_type<int, double, std::plus<> >::type ).name() << std::endl;
	
	
	return 0;
}
