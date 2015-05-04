generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_MPI)
  add_executable(libgeodecomp_examples_redblackgaussseidel ${SOURCES})
  set_target_properties(libgeodecomp_examples_redblackgaussseidel PROPERTIES OUTPUT_NAME redblackgaussseidel)
  target_link_libraries(libgeodecomp_examples_redblackgaussseidel ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
