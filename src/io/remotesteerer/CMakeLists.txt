lgd_generate_sourcelists("./")

if (WITH_THREADS AND WITH_BOOST_ASIO)
  add_subdirectory(test/parallel_mpi_1)
  add_subdirectory(test/parallel_mpi_2)
  add_subdirectory(test/parallel_mpi_4)
  add_subdirectory(test/unit)
endif()
