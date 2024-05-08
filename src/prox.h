#pragma once

// #include <RcppArmadillo.h>
#include <armadillo>

enum class ProxMethod
{
  stack = 0,
  pava  = 1
};

void
prox_pava(arma::vec& y, const arma::vec& lambda, bool nonneg = true);

arma::vec
prox_pava_conv(std::vector<double>& x, const std::vector<double>& lambda, bool nonneg = true);

void
prox_stack(arma::vec& x, const arma::vec& lambda, bool nonneg = true);

arma::vec
prox_stack_conv(std::vector<double>& x, const std::vector<double>& lambda, bool nonneg = true);

arma::mat
prox(const arma::mat& beta,
     const arma::vec& lambda,
     const ProxMethod prox_method,
     bool nonneg = true);
