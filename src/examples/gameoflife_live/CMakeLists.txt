generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_VISIT)
  add_executable(libgeodecomp_examples_gameoflife_live ${SOURCES})
  set_target_properties(libgeodecomp_examples_gameoflife_live PROPERTIES OUTPUT_NAME gameoflife_live)
  target_link_libraries(libgeodecomp_examples_gameoflife_live ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
