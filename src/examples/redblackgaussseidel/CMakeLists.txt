generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_MPI)
  add_executable(redblackgaussseidel ${SOURCES})
  target_link_libraries(redblackgaussseidel ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
