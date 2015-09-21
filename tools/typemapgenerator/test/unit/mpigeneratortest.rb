require 'test/unit'
load 'mpigenerator.rb'
load 'mpiparser.rb'

class TestMPIGenerator < Test::Unit::TestCase
  def setup
    @generator = MPIGenerator.new("./")
    @parser = MPIParser.new("./test/fixtures/doc/xml/")
    @classes = %w{Wheel Tire Rim}
    @opts = @parser.resolve_forest(@classes)
  end

  def test_generate_single_map
    classes = %w{Engine}
    resolved_classes = @parser.resolve_forest(classes).resolved_classes

    method_definition =
      @generator.generate_single_map(classes[0],
                                     resolved_classes[classes[0]],
                                     @opts.resolved_parents[classes[0]])
    expected_def = File.read("./test/fixtures/references/generatemapengine.cpp")

    assert_equal(expected_def, method_definition)
  end

  def test_generate_header
    expected_header = File.read("./test/fixtures/references/typemaps.h")
    actual_header = @generator.generate_header(@opts)
    # avoid comparing absolute pathnames (beneficial as the tests have to run in various locations)
    expected_header.gsub!(/(#include <).*\/(\w+\.h>)/) { |m| $1+$2 }
    actual_header.gsub!(  /(#include <).*\/(\w+\.h>)/) { |m| $1+$2 }

    assert_equal(expected_header, actual_header)
  end

  def test_generate_forest
    expected =
      [@generator.generate_header(@opts),
       @generator.generate_source(@opts)]
    assert_equal(expected,
                 @generator.generate_forest(@opts))
  end

  def test_initialize
    generator = MPIGenerator.new("foo")
    # This should fail, as there are no templates in the directory
    # "foo" (and if everything is OK, initialize() should have set
    # this as the default dir).
    assert_raise(Errno::ENOTDIR, Errno::ENOENT) do
      generator.generate_header(@opts)
    end
  end

  def test_generate_source
    @generator = MPIGenerator.new("./", "FooBar")
    actual_source = @generator.generate_source(@opts)
    actual_header = @generator.generate_header(@opts)
    assert_match(/^namespace FooBar \{/, actual_source)
    assert_match(/^namespace FooBar \{/, actual_header)
  end
end
