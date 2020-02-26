#include <armadillo>
#include "math_helper.h"
#include "widgets.h"

using namespace arma;

int main() {

	Stopwatch sw;
	int sz = 90000;

	vec a = ones(sz);

	vec b = {0.1};

	sw.run();

	auto c = gauss(b, a.t(), 0.05);


	sw.report();

	auto d = gauss(a, b, 0.05).t();
	
	sw.report();

	return 0;
}
