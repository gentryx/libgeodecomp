#generate_sourcelists("./")
set(RELATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/")
include(auto.cmake)

enable_language(Fortran)

#message("------->>>>  SOURCES= ${SOURCES}")

if(WITH_HPX)
  add_hpx_executable(gameoflife_adcirc
                     SOURCES ${SOURCES})

  target_link_libraries(gameoflife_adcirc_exe ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
