#include <join.h>
#include <iostream>
#include <armadillo>

using namespace arma;

int main() {
	uword sz = 3;	

	mat m1 = zeros(sz,1);
	mat m2 = ones(sz,2);
	mat m3 = randu(sz,3);

	mat M = join( { {m1,m2,m3}, {m3,m2,m1} } );
	M.print();
	std::cout << std::endl;

	sp_mat n1 = sprandu(sz,1,0.3);
	sp_mat n2 = sprandu(sz,2,0.3);
	sp_mat n3 = sprandu(sz,3,0.3);

	mat R = mat{join_r({n1,n2,n3})};
	R.print();
	std::cout << std::endl;
	
	sp_mat N = join( { {n1,n2,n3}, {n3,n2,n1} } );
	N.print();
	std::cout << std::endl;

	return 0;
}
