generate_sourcelists("./")

if (FEATURE_THREADS)
  add_subdirectory(test/parallel_mpi_4)
  add_subdirectory(test/unit)
endif()
