#include <armadillo>
#include "math_helper.h"
#include "widgets.h"

using namespace arma;

int main(int, char**argv) {

	Stopwatch sw;

	int m, n;
	readargs(argv, m, n);

	vec a = ones(m);
	rowvec b = ones<rowvec>(n);
	mat c(m,n);
	mat d;

	sw.run();
	c = repmat(a, 1, n) - repmat(b, m, 1);
	sw.report();
	sw.reset();

	d = c;

	sw.run();
	c = repelem(a, 1, n) - repelem(b, m, 1);
	sw.report();
	sw.reset();
	std::cout << accu(square(d-c)) << std::endl;

	sw.run();
	c = repmat(a,1,n).eval().each_row() - b;
	sw.report();
	sw.reset();
	std::cout << accu(square(d-c)) << std::endl;

	sw.run();
	c = a*ones(1,n) - ones(m,1)*b;
	sw.report();
	sw.reset();
	std::cout << accu(square(d-c)) << std::endl;

	sw.run();
	c = bcast_minus(a,b);
	sw.report();
	sw.reset();
	std::cout << accu(square(d-c)) << std::endl;

	return 0;
}
