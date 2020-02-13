#include "SIAM.h"
#include "newtonroot.h"

using namespace arma;

SIAM::SIAM(
		vec const& bath_,
		vec const& cpl_,
		double const& U_,
		uword const& n_elec_
): 
	bath(bath_), cpl(cpl_), U(U_), n_elec(n_elec_)
{
}
