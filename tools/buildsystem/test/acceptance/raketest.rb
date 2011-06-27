require 'test/common'

class RakeTest < Test::Unit::TestCase
  include UtilityFunctions

  def test_normal_build
    targets = %w{libsuperlib.so testlib/test/unit/test testexe/testexe}
    targets.each do |file|
      assert_file_exists(@opts.builddir + file)
    end
    assert_match(/testexe rulez/, 
                 sh(@opts.builddir + "testexe/testexe"))
    assert_match(/EchoTest\:\:testSimple/, @log)
  end  

  def test_build_twice_doesnt_compile_twice
    log = build
    assert_no_match(/Linking CXX executable/, log)
    assert_match(/EchoTest\:\:testSimple/, log)
  end
  
  def test_modified_header_causes_including_exes_to_be_updated
    assert_match(/ping/, sh(@opts.builddir + "testexe/testexe"))
    assert_no_match(/pong/, sh(@opts.builddir + "testexe/testexe"))

    patch("src_patched_header")
    log = build
    compiles = log.split("\n").find_all do |l| 
      l =~ /Building CXX object/ 
    end

    # check that not too many files are compiled
    assert_match(/Building CXX object.*echo\.cpp\.o/, log)
    assert_match(/Building CXX object.*typemaps\.cpp\.o/, log)
    assert_match(/Building CXX object.*main\.cpp\.o/, log)
    assert_equal(3, compiles.size)
    assert_no_match(/ping/, sh(@opts.builddir + "testexe/testexe"))
    assert_match(/pong/, sh(@opts.builddir + "testexe/testexe"))
  end

  def test_pri_files_are_updated
    pri = File.read(@opts.srcdir + "testlib/auto.cmake")
    assert_no_match(/greeter.h/, pri)

    patch("src_additional_header-only_class")
    build

    pri = File.read(@opts.srcdir + "testlib/auto.cmake")
    assert_match(/greeter.h/, pri)
  end

  def test_library_update_after_adding_class
    lib = @opts.builddir + "libsuperlib.so"
    assert_no_match(/Incrementer\:\:inc/, `objdump -t #{lib} | c++filt`)

    patch("src_additional_class")
    build

    assert_match(/Incrementer\:\:inc/, `objdump -t #{lib} | c++filt`)
  end

  def test_executable_update_after_patching_class
    exe = @opts.builddir + "testexe/testexe"
    assert_no_match(/echo is/, sh(@opts.builddir + "testexe/testexe"))

    patch("src_patched_class")
    build
    assert_match(/echo is/, sh(@opts.builddir + "testexe/testexe"))
  end

  def test_compiler_flags_are_used
    lines = @log.split("\n").find_all { |l| l =~ /mpic\+\+.*-c/ }
    assert_equal(5, lines.size)
    lines.each { |l| assert_match(/-DDOOMSDAY/, l) }
  end

  def test_additional_library_is_built_and_correct_compiler_is_used
    patch("src_additional_library")
    log = build
    lines = log.split("\n").find_all { |l| l =~ /mpic\+\+ / }

    assert_file_exists(@opts.builddir + "multiplier/libmultiplier.a")    
    # assert that GCC is used once (for compilation)
    assert_equal(1, lines.size)
  end

  def test_additional_library_is_tested_as_well_as_the_old_lib
    patch("src_additional_library")
    patch("src_additional_library_tests")
    log = build
    assert_match(/EchoTest\:\:testSimple/, log)
    assert_match(/MultiplierTest megalomania megalomania megalomania /, log)
  end

  def test_additional_library_is_tested_as_well_as_the_old_lib_iterative
    patch("src_additional_library")
    build
    patch("src_additional_library_tests")
    log = build
    assert_match(/EchoTest\:\:testSimple/, log)
    assert_match(/MultiplierTest megalomania megalomania megalomania /, log)
  end

  def test_additional_exe_is_built_and_runs
    exe = @opts.builddir + "powermonger/powermonger"
    assert_file_doesnt_exist(exe)

    patch("src_additional_exe")
    log = build
    compiles = log.split("\n").find_all { |l| l =~ /mpic\+\+/ }

    # check that only powermonger is compiled and linked
    assert_equal(2, compiles.size)
    assert_file_exists(exe)
    assert_match(/powermonger rulez/, sh(exe))    
  end

  def test_run_only_selected_test
    patch("src_additional_library")
    patch("src_additional_library_tests")
    log_complete = build
    log_selected = build("test", @opts.srcdir + "multiplier")

    assert_match(/EchoTest/, log_complete)
    assert_no_match(/EchoTest/, log_selected)

    assert_match(/MultiplierTest/, log_complete)
    assert_match(/MultiplierTest/, log_selected)
  end

  def test_doc_generation
    patch("src_additional_library")
    log = build("doc")

    %w{../doc/html/classEcho.html ../doc/xml/classEcho.xml}.each do |w|
      assert_file_exists(@opts.srcdir + w)
    end
    assert_match(/doxygen doxygen\.conf/, log)
  end

  def test_typemaps_generation
    files = %w{mpilayer/typemaps.h mpilayer/typemaps.cpp}
    files.each do |w|
      assert_no_match(/Echo/,  File.read(@opts.srcdir + w))
    end

    patch("src_patched_header")
    build

    files.each do |w|
      assert_match(/Echo/,  File.read(@opts.srcdir + w))
    end
  end

  def test_typemaps_creation_for_multiple_classes
    files = %w{mpilayer/typemaps.h mpilayer/typemaps.cpp}
    files.each do |w|
      assert_no_match(/Echo/,  File.read(@opts.srcdir + w))
      assert_no_match(/Greeter/,  File.read(@opts.srcdir + w))
    end

    patch("src_patched_header")
    patch("src_additional_header-only_class")
    log = build

    files.each do |w|
      assert_match(/Echo/,  File.read(@opts.srcdir + w))
      assert_match(/Greeter/,  File.read(@opts.srcdir + w))
    end
    assert_match(/^regenerating Typemaps$/, log)
  end

  def test_typemaps_are_only_updated_when_neccessary
    log = build
    assert_no_match(/^regenerating Typemaps$/, log)
    
    patch("src_patched_header")
    log = build
    assert_match(/^regenerating Typemaps$/, log)
    
    patch("src_patched_class")
    log = build
    assert_no_match(/^regenerating Typemaps$/, log)
    
    log = build
    assert_no_match(/^regenerating Typemaps$/, log)
    
    patch("src_additional_header-only_class")
    log = build
    assert_match(/^regenerating Typemaps$/, log)
    
    patch("src_additional_library")
    log = build
    assert_no_match(/^regenerating Typemaps$/, log)
    
    patch("src_additional_library_tests")
    log = build
    assert_no_match(/^regenerating Typemaps$/, log)
  end

  def test_parallel_tests_are_built_and_run
    patch("src_mpi_tests")
    log = build

    assert_match(/size is 9/, log)
    assert_match(/sum is 511/, log)
  end

  def test_faulty_compilation_stops_test_process
    patch("src_faulty_test")
    log = build("test", @opts.srcdir, false)
    assert_match(/rake aborted/, log)
    assert_no_match(/Running.*test/, log)
  end

  def test_clean_does_clean_thoroughly
    before = @file_list
    log = build("clean")
    after = file_list

    # the build dir itself may continue to exist after clean, but its
    # contents should be gone.
    after.delete((@tmp_dir + "build").to_s)
    # config.{yaml|cmake} should only be deleted by "rake distclean".
    after.delete((@tmp_dir + "src" + "conf.yaml").to_s)
    after.delete((@tmp_dir + "src" + "conf.cmake").to_s)

    assert_equal(before, after)
  end

  def test_distclean_does_clean_thoroughly
    before = @file_list
    log = build("distclean")
    after = file_list
    # the build dir itself may continue to exist after clean, but its
    # contents should be gone.
    after.delete((@tmp_dir + "build").to_s)

    assert_equal(before, after)
  end

  def test_install
    build "install"
    files = Dir.glob(@install_dir + "**" + "*").map do |f|
      Pathname.new(f).relative_path_from(@install_dir)
    end
    files.map! { |f| f.to_s }
    expected = %w(include include/testpackage include/testpackage/mpilayer include/testpackage/mpilayer/typemaps.h include/testpackage/testlib include/testpackage/testlib/echo.h include/testpackage/testlib/testlib.h lib lib/libsuperlib.so)

    assert_equal(expected, files.sort)
  end

  def test_builddir_with_spaces
    teardown
    setup("src", "", "raketest foobar")
  end

  def test_enable_mpi_tests
    teardown
    setup("src")
    patch("src_mpi_tests")
    log = build
    assert_match(/bin\/mpiexec -np 9/, log) 
    assert_match(/size is 9/, log) 
  end

  def test_disable_mpi_tests
    teardown
    setup("src", "--no-mpi")
    patch("src_mpi_tests")
    log = build
    assert_no_match(/usr\/bin\/mpiexec -np 9/, log) 
    assert_no_match(/size is 9/, log) 
  end

  def test_cleaning_additional_files_from_configure
    patch("src_additional_files")
    assert_file_exists(@opts.srcdir + "foobar")
    assert_file_exists(@opts.srcdir + "blah" + "foobar")
    build("distclean")
    assert_file_doesnt_exist(@opts.srcdir + "foobar")
    assert_file_doesnt_exist(@opts.srcdir + "blah" + "foobar")
  end
end
