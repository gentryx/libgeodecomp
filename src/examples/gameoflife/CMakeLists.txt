generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(FEATURE_MPI)
  add_executable(gameoflife main.cpp)
  target_link_libraries(gameoflife ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif(FEATURE_MPI)

if(FEATURE_HPX)
  add_hpx_executable(
    gameoflife_hpx
    SOURCES main_hpx.cpp
    DEPENDENCIES ${LOCAL_LIBGEODECOMP_LINK_LIB}
  )
endif()
