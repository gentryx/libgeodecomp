generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_SILO)
  add_executable(libgeodecomp_examples_chromatography ${SOURCES})
  set_target_properties(libgeodecomp_examples_chromatography PROPERTIES OUTPUT_NAME chromatography)
  target_link_libraries(libgeodecomp_examples_chromatography ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif(WITH_SILO)
