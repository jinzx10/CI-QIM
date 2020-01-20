#include <dc.h>
#include <armadillo>

int main() {
	int sz = 5;
	arma::mat B = arma::randn(sz, sz);
	arma::mat A, r, vecs;
	arma::qr(A, r, B);
	B.randn();
	arma::qr(vecs, r, B);

	////////////////////////////////////////////////////////////
	//						type 1
	////////////////////////////////////////////////////////////
	// column
	arma::mat smart1_col = det12(A, arma::uvec{0}, vecs);
	arma::mat dumb1_col = arma::zeros(sz);

	arma::mat tmp = A;
	for (int i = 0; i != sz; ++i) {
		tmp.col(0) = vecs.col(i);
		dumb1_col(i) = arma::det(tmp);
	}

	std::cout << "type 1: col" << std::endl;
	std::cout << "dumb" << std::endl;
	dumb1_col.print();
	std::cout << std::endl;
	std::cout << "smart" << std::endl;
	smart1_col.print();
	std::cout << std::endl;

	// row
	arma::mat smart1_row = det12(A, arma::uvec{0}, vecs, 'r');
	arma::mat dumb1_row = arma::zeros(sz);
	tmp = A;
	for (int i = 0; i != sz; ++i) {
		tmp.row(0) = vecs.row(i);
		dumb1_row(i) = arma::det(tmp);
	}

	std::cout << "type 1: row" << std::endl;
	std::cout << "dumb" << std::endl;
	dumb1_row.print();
	std::cout << std::endl;
	std::cout << "smart" << std::endl;
	smart1_row.print();
	std::cout << std::endl;

	////////////////////////////////////////////////////////////
	//						type 2
	////////////////////////////////////////////////////////////
	// column
	arma::mat smart2_col = det12(A, arma::regspace<arma::uvec>(0, sz-1), vecs.col(0));
	arma::mat dumb2_col = arma::zeros(sz);
	for (int i = 0; i != sz; ++i) {
		tmp = A;
		tmp.col(i) = vecs.col(0);
		dumb2_col(i) = arma::det(tmp);
	}
	std::cout << "type 2: col" << std::endl;
	std::cout << "smart" << std::endl;
	smart2_col.print();
	std::cout << std::endl;
	std::cout << "dumb" << std::endl;
	dumb2_col.print();
	std::cout << std::endl;

	// row
	arma::mat smart2_row = det12(A, arma::regspace<arma::uvec>(0, sz-1), vecs.row(0),'r');
	arma::mat dumb2_row = arma::zeros(sz);
	for (int i = 0; i != sz; ++i) {
		tmp = A;
		tmp.row(i) = vecs.row(0);
		dumb2_row(i) = arma::det(tmp);
	}
	std::cout << "type 2: row" << std::endl;
	std::cout << "smart" << std::endl;
	smart2_row.print();
	std::cout << std::endl;
	std::cout << "dumb" << std::endl;
	dumb2_row.print();
	std::cout << std::endl;

	////////////////////////////////////////////////////////////
	//						type 3
	////////////////////////////////////////////////////////////
	// column
	arma::vec u = arma::randu(sz);
	arma::vec v = arma::randu(sz);
	arma::mat smart3_col = det3(A, 0, u, arma::regspace<arma::uvec>(1, sz-1), v);
	arma::mat dumb3_col = arma::zeros(sz-1);
	for (int i = 1; i != sz; ++i) {
		tmp = A;
		tmp.col(0) = u;
		tmp.col(i) = v;
		dumb3_col(i-1) = arma::det(tmp);
	}
	std::cout << "type 3: col" << std::endl;
	std::cout << "smart" << std::endl;
	smart3_col.print();
	std::cout << std::endl;
	std::cout << "dumb" << std::endl;
	dumb3_col.print();
	std::cout << std::endl;

	// row
	arma::mat smart3_row = det3(A, 0, u.t(), arma::regspace<arma::uvec>(1, sz-1), v.t(), 'r');
	arma::mat dumb3_row = arma::zeros(sz-1);
	for (int i = 1; i != sz; ++i) {
		tmp = A;
		tmp.row(0) = u.t();
		tmp.row(i) = v.t();
		dumb3_row(i-1) = arma::det(tmp);
	}
	std::cout << "type 3: row" << std::endl;
	std::cout << "smart" << std::endl;
	smart3_row.print();
	std::cout << std::endl;
	std::cout << "dumb" << std::endl;
	dumb3_row.print();
	std::cout << std::endl;

	return 0;
}
