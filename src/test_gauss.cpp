#include "../include/gauss.h"

using namespace arma;

int main() {

	vec a = randu(3,1);
	rowvec b = randu(1,5);

	std::cout << typeid(a).name() << std::endl;
	std::cout << typeid(b).name() << std::endl;
	mat c = gauss(a,b,0.1);

	a.print();
	b.print();
	c.print();

	gauss(a,b,0.1).print();

	return 0;
}
