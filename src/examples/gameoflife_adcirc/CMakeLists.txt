lgd_generate_sourcelists("./")
set(RELATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/")
include(auto.cmake)

if(WITH_HPX AND WITH_FORTRAN AND WITH_SILO)
  add_executable(libgeodecomp_examples_gameoflife_adcirc ${SOURCES})
  set_target_properties(libgeodecomp_examples_gameoflife_adcirc PROPERTIES OUTPUT_NAME gameoflife_adcirc)
  target_link_libraries(libgeodecomp_examples_gameoflife_adcirc ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
