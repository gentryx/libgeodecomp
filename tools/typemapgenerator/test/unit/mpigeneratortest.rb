require 'test/unit'
require 'mpigenerator'
require 'mpiparser'

class TestMPIGenerator < Test::Unit::TestCase
  def setup
    @generator = MPIGenerator.new("./")
    @parser = MPIParser.new("./test/fixtures/doc/xml/")
    @classes = %w{Wheel Tire Rim}
    @resolved_classes, @resolved_parents, @datatype_map, @topological_class_sortation, @headers =
      @parser.resolve_forest(@classes)
  end

  def test_generate_single_map
    classes = %w{Engine}
    resolved_classes = @parser.resolve_forest(classes)[0]

    method_definition =
      @generator.generate_single_map(classes[0], resolved_classes[classes[0]], @resolved_parents[classes[0]])
    expected_def = File.read("./test/fixtures/references/generatemapengine.cpp")

    assert_equal(expected_def, method_definition)
  end

  def test_generate_header
    expected_header = File.read("./test/fixtures/references/typemaps.h")
    actual_header = @generator.generate_header(@topological_class_sortation,
                                               @datatype_map,
                                               @resolved_classes,
                                               @resolved_parents,
                                               @headers)
    # avoid comparing absolute pathnames (beneficial as the tests have to run in various locations)
    expected_header.gsub!(/(#include <).*\/(\w+\.h>)/) { |m| $1+$2 }
    actual_header.gsub!(  /(#include <).*\/(\w+\.h>)/) { |m| $1+$2 }

    assert_equal(expected_header, actual_header)
  end

  def test_generate_source
    source_pattern = File.read("./test/fixtures/references/typemaps.cpp")
    actual_source = @generator.generate_source(@topological_class_sortation,
                                               @datatype_map,
                                               @resolved_classes)
    assert_match(/#{source_pattern}/m, actual_source)
  end

  def test_generate_forest
    expected =
      [@generator.generate_header(@topological_class_sortation,
                                  @datatype_map,
                                  @resolved_classes,
                                  @resolved_parents,
                                  @headers),
       @generator.generate_source(@topological_class_sortation,
                                  @datatype_map,
                                  @resolved_classes,
                                  @resolved_parents)]
    assert_equal(expected,
                 @generator.generate_forest(@resolved_classes,
                                            @resolved_parents,
                                            @datatype_map,
                                            @topological_class_sortation,
                                            @headers))
  end

  def test_initialize
    generator = MPIGenerator.new("foo")
    # This should fail, as there are no templates in the directory
    # "foo" (and if everything is OK, initialize() should have set
    # this as the default dir).
    assert_raise(Errno::ENOTDIR, Errno::ENOENT) do
      generator.generate_header(@topological_class_sortation,
                                @datatype_map,
                                @resolved_classes,
                                @resolved_parents,
                                @headers)
    end
  end

  def test_generate_source
    @generator = MPIGenerator.new("./", "FooBar")
    actual_source = @generator.generate_source(@topological_class_sortation,
                                               @datatype_map,
                                               @resolved_classes,
                                               @resolved_parents)
    actual_header = @generator.generate_header(@topological_class_sortation,
                                               @datatype_map,
                                               @resolved_classes,
                                               @resolved_parents,
                                               @headers)
    assert_match(/^namespace FooBar \{/, actual_source)
    assert_match(/^namespace FooBar \{/, actual_header)
  end
end
