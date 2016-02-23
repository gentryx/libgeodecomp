lgd_generate_sourcelists("./")
set(RELATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/")
include(auto.cmake)

if(WITH_HPX)
  add_executable(libgeodecomp_examples_gameoflife_hpx ${SOURCES})
  set_target_properties(libgeodecomp_examples_gameoflife_hpx PROPERTIES OUTPUT_NAME gameoflife_hpx)
  target_link_libraries(libgeodecomp_examples_gameoflife_hpx ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
