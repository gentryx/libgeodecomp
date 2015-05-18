generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_CPP14)
  add_executable(spmvmvectorized ${SOURCES})
  target_link_libraries(spmvmvectorized ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
