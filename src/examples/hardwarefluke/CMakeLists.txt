generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(libgeodecomp_examples_hardwarefluke ${SOURCES})
set_target_properties(libgeodecomp_examples_hardwarefluke PROPERTIES OUTPUT_NAME hardwarefluke)
target_link_libraries(libgeodecomp_examples_hardwarefluke ${LOCAL_LIBGEODECOMP_LINK_LIB})
