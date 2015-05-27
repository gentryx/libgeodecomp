lgd_generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_MPI AND WITH_VISIT AND WITH_THREADS AND WITH_BOOST_ASIO)
  add_executable(libgeodecomp_examples_gameoflife3d ${SOURCES})
  set_target_properties(libgeodecomp_examples_gameoflife3d PROPERTIES OUTPUT_NAME gameoflife3d)
  target_link_libraries(libgeodecomp_examples_gameoflife3d ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
