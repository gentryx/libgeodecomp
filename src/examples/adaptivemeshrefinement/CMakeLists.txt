generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(adaptivemeshrefinement ${SOURCES})
target_link_libraries(adaptivemeshrefinement ${LOCAL_LIBGEODECOMP_LINK_LIB})
