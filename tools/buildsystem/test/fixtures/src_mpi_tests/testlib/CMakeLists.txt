add_subdirectory(test/unit)
add_subdirectory(test/parallel_mpi_9)

include(auto.cmake)

install(FILES ${HEADERS} DESTINATION "${INCLUDE_INSTALL_DIR}")
