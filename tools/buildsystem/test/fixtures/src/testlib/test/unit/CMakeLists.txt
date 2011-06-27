include(auto.cmake)

include_directories (${DEMO_SOURCE_DIR}/../lib/cxxtest)
add_executable(testlib_test_unit ${SOURCES})
set_target_properties(testlib_test_unit PROPERTIES OUTPUT_NAME test)
target_link_libraries (testlib_test_unit superlib)
