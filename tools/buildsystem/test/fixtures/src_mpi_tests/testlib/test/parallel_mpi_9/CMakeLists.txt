include(auto.cmake)

if(FEATURE_MPI_TEST)
  include_directories(${DEMO_SOURCE_DIR}/../lib/cxxtest)
  add_executable(testlib_test_parallel_mpi_9 ${SOURCES})
  set_target_properties(testlib_test_parallel_mpi_9 PROPERTIES OUTPUT_NAME test)
endif(FEATURE_MPI_TEST)
