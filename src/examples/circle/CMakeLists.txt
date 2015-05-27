lgd_generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(libgeodecomp_examples_circle ${SOURCES})
set_target_properties(libgeodecomp_examples_circle PROPERTIES OUTPUT_NAME circle)
target_link_libraries(libgeodecomp_examples_circle ${LOCAL_LIBGEODECOMP_LINK_LIB})
