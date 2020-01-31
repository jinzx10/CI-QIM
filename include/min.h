#ifndef __MININUM_H__
#define __MININUM_H__

inline double min(double const& i) {
	return i;
}

template <typename ...Ts>
double min(double const& i, Ts const& ...args) {
    double tmp = min(args...);
    return ( i < tmp ) ? i : tmp;
}

#endif
