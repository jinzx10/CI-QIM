/* numerical gradient of real functions by finite difference
 */

#ifndef __GRADIENT_H__
#define __GRADIENT_H__

#include <functional>

inline std::function<double(double)> grad(std::function<double(double)> const& f) {
	double delta = 0.001;
	return [=] (double x) -> double {
		return ( -f(x-3.0*delta)/60.0 + 3.0*f(x-2.0*delta)/20.0 - 3.0*f(x-delta)/4.0 +
				f(x+3.0*delta)/60.0 - 3.0*f(x+2.0*delta)/20.0 + 3.0*f(x+delta)/4.0 ) 
			/ delta;
	};
}

template <typename Vector>
std::function<double(Vector)> gradi(std::function<double(Vector)> const& f, size_t const& i) {
	return [=] (Vector const& v) -> double {
		std::function<double(double)> g = [=, v=v] (double const& x) mutable {
			v[i] = x;
			return f(v);
		};
		return grad(g)(v[i]); 
	};
}

template <typename Vector>
std::function<Vector(Vector)> grad(std::function<double(Vector)> const& f) {
	return [=] (Vector const& x) -> Vector {
		Vector df = x;
		for (size_t i = 0; i != x.size(); ++i) {
			df[i] = gradi(f,i)(x);
		}
		return df;
	};
}

#endif
