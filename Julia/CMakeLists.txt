cmake_minimum_required(VERSION 3.10) # Set the minimum required version of cmake
project(CppSLOPE)                    # Define the project name

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the Julia package required
# list(APPEND CMAKE_PREFIX_PATH "/Users/jakeroth/.julia/artifacts/2ce0502d68a1c277361b766c5050582bf167ad43")
# Find the Julia executable
find_program(JULIA_EXECUTABLE NAMES julia)

if(NOT JULIA_EXECUTABLE)
  message(FATAL_ERROR "Julia not found.")
endif()

# Execute Julia command to get the CxxWrap prefix path
execute_process(
  COMMAND ${JULIA_EXECUTABLE} -e "using CxxWrap; println(CxxWrap.prefix_path())"
  OUTPUT_VARIABLE CXXWRAP_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Output the retrieved path for verification
message(STATUS "CxxWrap prefix path: ${CXXWRAP_PREFIX_PATH}")

# Append the CxxWrap prefix path to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH ${CXXWRAP_PREFIX_PATH})

# Include the JlCxx and Armadillo libraries
find_package(JlCxx REQUIRED)
find_package(Armadillo REQUIRED)
if(ARMADILLO_FOUND)
    include_directories(${ARMADILLO_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Armadillo not found")
endif()
message(STATUS "Armadillo include dir: ${ARMADILLO_INCLUDE_DIRS}")

# Add your source files
set(SOURCE_FILES
    ../src/prox.cpp
    ../src/JcppExports.cpp
)

# Create a shared library called CppSLOPE
add_library(CppSLOPE SHARED ${SOURCE_FILES})

# Link JlCxx and Armadillo libraries to SLOPE
target_link_libraries(CppSLOPE JlCxx::cxxwrap_julia ${ARMADILLO_LIBRARIES})
