#include <armadillo>
#include <jlcxx/jlcxx.hpp>
#include <vector>
#include "prox.h"  // Header that declares evaluateProx and other functions

// Function to convert a std::vector<double> to arma::vec
arma::vec std_vector_to_arma_vec(const std::vector<double>& x) {
    arma::vec arma_x = arma::conv_to<arma::vec>::from(x);
    return arma_x;
}

// Function to convert arma::vec to std::vector<double> (for full vector retrieval)
std::vector<double> arma_vec_to_std_vector(const arma::vec& v) {
    return std::vector<double>(v.begin(), v.end());
}

// Function to use std::vector in prox_pava
arma::vec prox_pava_conv(std::vector<double>& x, const std::vector<double>& lambda, bool nonneg) {
    // Convert std::vector to arma::vec
    arma::vec arma_x = arma::conv_to<arma::vec>::from(x);
    arma::vec arma_lambda = arma::conv_to<arma::vec>::from(lambda);

    // Call prox_pava with arma::vec
    prox_pava(arma_x, arma_lambda, nonneg);
    return arma_x;
}

// Function to use std::vector in prox_stack
arma::vec prox_stack_conv(std::vector<double>& x, const std::vector<double>& lambda, bool nonneg) {
    // Convert std::vector to arma::vec
    arma::vec arma_x = arma::conv_to<arma::vec>::from(x);
    arma::vec arma_lambda = arma::conv_to<arma::vec>::from(lambda);

    // Call prox_pava with arma::vec
    prox_stack(arma_x, arma_lambda, nonneg);
    return arma_x;
}

// Define the module's interface for Julia
JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    // Register Armadillo types with Julia
    // mod.add_type<arma::mat>("Mat")
    //   .constructor<int, int>();
    mod.add_type<arma::vec>("Vec")
      .constructor<int>();
    // mod.method("vector_to_arma_vec", &vector_to_arma_vec);
    // mod.method("vector_to_arma_mat", &vector_to_arma_mat);

    // Register the ProxMethod enum
    // mod.add_bits<ProxMethod>("ProxMethod");
    // mod.set_const("stack", ProxMethod::stack);
    // mod.set_const("pava", ProxMethod::pava);

    // Provide a Julia interface to the prox functions
    mod.method("prox_pava_conv", &prox_pava_conv);
    mod.method("prox_stack_conv", &prox_stack_conv);
    mod.method("prox_pava", &prox_pava);
    mod.method("prox_stack", &prox_stack);
    mod.method("std2arma", &std_vector_to_arma_vec);
    mod.method("arma2std", &arma_vec_to_std_vector);
    
    // mod.method("prox_stack_inplace", [](std::vector<double>& x, const std::vector<double>& lambda) {
    //     prox_stack_inplace(x, lambda);  // Call prox_stack, which modifies x in-place
    // });
    // mod.method("prox_pava_inplace", [](std::vector<double>& x, const std::vector<double>& lambda) {
    //     prox_pava_inplace(x, lambda);  // Call prox_pava, which modifies x in-place
    // });
    
    // prox convenience
    // mod.method("prox", [](const arma::mat& beta, const arma::vec& lambda, ProxMethod prox_method) -> arma::mat {
    //     return prox(beta, lambda, prox_method);
    // });
}