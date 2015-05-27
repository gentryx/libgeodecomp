lgd_generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)


if(WITH_SILO)
  add_executable(libgeodecomp_examples_adaptivemeshrefinement ${SOURCES})
  set_target_properties(libgeodecomp_examples_adaptivemeshrefinement PROPERTIES OUTPUT_NAME adaptivemeshrefinement)
  target_link_libraries(libgeodecomp_examples_adaptivemeshrefinement ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
