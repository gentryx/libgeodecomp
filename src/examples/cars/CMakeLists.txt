lgd_generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_MPI AND WITH_VISIT)
  add_executable(libgeodecomp_examples_cars ${SOURCES})
  set_target_properties(libgeodecomp_examples_cars PROPERTIES OUTPUT_NAME cars)
  target_link_libraries(libgeodecomp_examples_cars ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
