generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_MPI)
  add_executable(globalreductions ${SOURCES})
  target_link_libraries(globalreductions ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
