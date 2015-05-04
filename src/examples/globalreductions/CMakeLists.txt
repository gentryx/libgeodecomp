generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_MPI)
  add_executable(libgeodecomp_examples_globalreductions ${SOURCES})
  set_target_properties(libgeodecomp_examples_globalreductions PROPERTIES OUTPUT_NAME globalreductions)
  target_link_libraries(libgeodecomp_examples_globalreductions ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
