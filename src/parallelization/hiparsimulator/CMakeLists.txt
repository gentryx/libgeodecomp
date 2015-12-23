lgd_generate_sourcelists("./")

add_subdirectory(test/parallel_mpi_1)
add_subdirectory(test/parallel_mpi_4)

lgd_escape_kernel("escapedopenclkernel.h" "openclkernel.cl")

# add_custom_target(hiparsimulator_code_generation echo "HiParSimulator code generation done."
#   DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/escapedopenclkernel.h")
# add_dependencies(code_generation hiparsimulator_code_generation)
