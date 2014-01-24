generate_sourcelists("./")

if (WITH_THREADS)
  add_subdirectory(test/parallel_mpi_2)
  add_subdirectory(test/parallel_mpi_4)
  add_subdirectory(test/unit)
endif()
