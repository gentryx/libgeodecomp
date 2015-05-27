lgd_generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_SILO)
  add_executable(libgeodecomp_examples_voronoi ${SOURCES})
  set_target_properties(libgeodecomp_examples_voronoi PROPERTIES OUTPUT_NAME voronoi)
  target_link_libraries(libgeodecomp_examples_voronoi ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
