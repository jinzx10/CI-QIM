#ifndef __TEMPLATE_HELPER_H__
#define __TEMPLATE_HELPER_H__

#include <type_traits>

template <typename ...>
using void_t = void;

template <typename F, typename ...Args>
struct is_valid_call
{
    typedef char yes;
    typedef char no[2];

	template <typename F1, typename ...Args1>
	using return_t = decltype( std::declval<F1>()( std::declval<Args1>()... ) );

    template <typename F1, typename ...Args1>
    static yes& test(return_t<F1, Args1...>* );

    template <typename F1, typename ...Args1>
    static no& test(...);

    static const bool value = sizeof(test<F, Args...>(nullptr)) == sizeof(yes);

	template <bool, typename, typename ...>
	struct try_return { using type = void; };
	
	template <typename F1, typename ...Args1>
	struct try_return<true, F1, Args1...> { using type = return_t<F1, Args...>; };

	using return_type = typename try_return<value, F, Args...>::type;

};


#endif
