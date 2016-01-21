lgd_generate_sourcelists("./")
set(RELATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/")
include(auto.cmake)

if(WITH_HPX AND WITH_FORTRAN)
  add_executable(gameoflife_adcirc ${SOURCES})
  target_link_libraries(gameoflife_adcirc ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
