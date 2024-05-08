using CxxWrap
module CppSLOPE
  using CxxWrap
  import CxxWrap: StdVector

  # Define a function that returns the path to the dynamic library
  function get_lib_path()
    # Ensure this exactly matches the library name and path
    return joinpath(@__DIR__, "build/libCppSLOPE.dylib")
  end

  # Correctly pass the function to @wrapmodule
  @wrapmodule(get_lib_path)  # Note the absence of ()

  # # Julia convenience wrapper for evaluateProx that handles vector conversions
  # function evaluateProx(z_in::AbstractVector{Tf}, w::AbstractVector{Tf}, epsilon::Tf, x_out::AbstractVector{Tf}, sorted_and_positive::Bool) where {Tf<:AbstractFloat}
  #   # Call the C++ function
  #   evaluateProx(StdVector{Tf}(z_in), StdVector{Tf}(w), epsilon, StdVector{Tf}(x_out), sorted_and_positive)
  # end

  function __init__()
    @initcxx
  end
end
# Check if the module is loaded
println("Module loaded: ", isdefined(Main, :CppSLOPE))


# Example usage of the prox function with enum
n = 10_000_000
k = Int(0.1*n)
x0 = 100rand(n)
lambda0 = [sort(rand(k), rev=true); zeros(n-k)]
x = CxxWrap.StdVector(zeros(Float64, n))
lambda = CxxWrap.StdVector(zeros(Float64, n))
x .= x0
lambda .= lambda0

@time xout_pava = CppSLOPE.prox_pava_conv(x, lambda);
@time xout_stack = CppSLOPE.prox_stack_conv(x, lambda);
@time xrec_pava = CppSLOPE.arma2std(xout_pava);
@time xrec_stack = CppSLOPE.arma2std(xout_pava);

x_arma_pava = CppSLOPE.std2arma(x)
x_arma_stack = CppSLOPE.std2arma(x)
lambda_arma = CppSLOPE.std2arma(lambda)
CppSLOPE.prox_pava(x_arma_pava, lambda_arma)
CppSLOPE.prox_stack(x_arma_stack, lambda_arma)
@time xrec_pava = CppSLOPE.arma2std(x_arma_pava)
@time xrec_stack = CppSLOPE.arma2std(x_arma_stack)

