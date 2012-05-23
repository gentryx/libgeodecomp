# list headers/source files in "auto.cmake"
function(generate_sourcelists relative_dir)
  get_filename_component(dir ${relative_dir} ABSOLUTE)
  # message("generate_sourcelists ${dir}")
 
  file(GLOB RAW_SOURCES "${dir}/*.cu" "${dir}/*.cpp")
  file(GLOB RAW_HEADERS "${dir}/*.h")

  if(RAW_SOURCES OR RAW_HEADERS)
    set(STRIPPED_SOURCES)
    set(STRIPPED_HEADERS)

    foreach(i ${RAW_SOURCES})
      get_filename_component(name ${i} NAME)
      list(APPEND STRIPPED_SOURCES ${name})
    endforeach(i)

    foreach(i ${RAW_HEADERS})
      get_filename_component(name ${i} NAME)
      list(APPEND STRIPPED_HEADERS ${name})
    endforeach(i)

    if(STRIPPED_SOURCES)
      list(SORT STRIPPED_SOURCES ${STRIPPED_SOURCES})
    endif(STRIPPED_SOURCES)
    
    if(STRIPPED_HEADERS)
      list(SORT STRIPPED_HEADERS ${STRIPPED_HEADERS})
    endif(STRIPPED_HEADERS)

    set(MY_AUTO "set(SOURCES \${SOURCES}\n")
    foreach(i ${STRIPPED_SOURCES})
      set(MY_AUTO "${MY_AUTO}  \${RELATIVE_PATH}${i}\n")
    endforeach(i)

    set(MY_AUTO "${MY_AUTO})\nset(HEADERS \${HEADERS}\n")
    foreach(i ${STRIPPED_HEADERS})
      set(MY_AUTO "${MY_AUTO}  \${RELATIVE_PATH}${i}\n")
    endforeach(i)
    set(MY_AUTO "${MY_AUTO})\n")

    # only actually write the file if it differs
    if(NOT EXISTS "${dir}/auto.cmake")
      set(regen_auto 1)
    endif(NOT EXISTS "${dir}/auto.cmake")

    if(EXISTS "${dir}/auto.cmake")
      file(READ "${dir}/auto.cmake" PREV_AUTO)
      string(COMPARE NOTEQUAL ${MY_AUTO} ${PREV_AUTO} regen_auto)
    endif(EXISTS "${dir}/auto.cmake")

    if (regen_auto)
      message("updating ${dir}//auto.cmake")
      file(WRITE "${dir}/auto.cmake" ${MY_AUTO})
    endif(regen_auto)
  endif(RAW_SOURCES OR RAW_HEADERS)
endfunction(generate_sourcelists)

# creates a string constant from a source file, handy for e.g.
# just-in-time compilation of OpenCL kernels
function(escape_kernel outfile infile)
  add_custom_command(
    OUTPUT "${outfile}"
    COMMAND cat "${CMAKE_CURRENT_SOURCE_DIR}/${infile}" | sed 's/"/\\\\"/g' | sed s/.*/\\"\\&\\"/ >"${CMAKE_CURRENT_SOURCE_DIR}/${outfile}"
    DEPENDS "${infile}"
    )
endfunction(escape_kernel)
