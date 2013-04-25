generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(FEATURE_VISIT)
  add_executable(gameoflife_live ${SOURCES})
  target_link_libraries(gameoflife_live ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
