lgd_generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_CPP14)
  add_executable(spmvm ${SOURCES})
  target_link_libraries(spmvm ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
