generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(jacobi ${SOURCES})
target_link_libraries(jacobi ${LOCAL_LIBGEODECOMP_LINK_LIB})
