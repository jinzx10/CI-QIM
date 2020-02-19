#include <iostream>
//#include <armadillo>
#include <type_traits>
#include "template_helper.h"
//#include "arma_helper.h"
//#include "math_helper.h"

//using namespace arma;

template <typename F1, typename ...Args1>
using return_t = decltype( std::declval<F1>()( std::declval<Args1>()... ) );

typedef char yes;
typedef char no[2];

template <typename F1, typename ...Args1>
static yes& test(decltype( std::declval<F1>()( std::declval<Args1>()... ) )*);
//static yes& test(return_t<F1, Args1...>*);

template <typename F1, typename ...Args1>
static no& test(...);


int main() {

	auto f = [] (double x, double y) { return x + y; };
	std::cout << (sizeof(test<decltype(f), double, double>(nullptr)) == sizeof(yes)) << std::endl;
	/*
	std::function<vec(vec,double)> minus = [] (arma::vec const& col, double const& x) -> arma::vec { return col-x; };

	return_t<decltype(minus), vec, double>* ptr;
	std::cout << typeid(ptr).name() << std::endl;

	std::cout << (sizeof(test<decltype(minus), vec, double>(nullptr)) == sizeof(yes)) << std::endl;
	*/
	/*
	std::cout << typeid(decltype(minus)).name() << std::endl;
	std::cout << typeid(minus).name() << std::endl;
	std::cout << is_valid_call<decltype(minus), vec, double>::value << std::endl;
	std::cout << typeid(is_valid_call<decltype(minus), vec, double>::return_type).name() << std::endl;
	*/
	
	return 0;
}
