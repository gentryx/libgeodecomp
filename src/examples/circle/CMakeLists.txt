generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(circle ${SOURCES})
target_link_libraries(circle ${LOCAL_LIBGEODECOMP_LINK_LIB})
