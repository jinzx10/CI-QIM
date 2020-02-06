#ifndef __DERIVATIVE_COUPLING_H__
#define __DERIVATIVE_COUPLING_H__

#include <armadillo>

void adj_phase(arma::mat const& vecs_old, arma::mat& vecs_new);

arma::mat ovl(arma::vec const& vec_do_, arma::mat const& vec_occ_, arma::vec const& vec_dv_, arma::mat const& vec_vir_, arma::vec const& vec_do, arma::mat const& vec_occ, arma::vec const& vec_dv, arma::mat const& vec_vir);
arma::mat ovl_dumb(arma::vec const& vec_do_, arma::mat const& vec_occ_, arma::vec const& vec_dv_, arma::mat const& vec_vir_, arma::vec const& vec_do, arma::mat const& vec_occ, arma::vec const& vec_dv, arma::mat const& vec_vir);

arma::vec det12(arma::mat const& A, arma::uvec const& idx, arma::mat const& vecs, char const& rc = 'c');
arma::vec det3(arma::mat const& A, arma::uword const& i, arma::mat const& u, arma::uvec const& idx, arma::mat const& v, char const& rc = 'c');



#endif
