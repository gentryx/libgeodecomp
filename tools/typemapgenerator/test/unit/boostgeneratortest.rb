require 'test/unit'
load 'boostgenerator.rb'
load 'mpiparser.rb'

class TestBoostGenerator < Test::Unit::TestCase
  def setup
    @generator = BoostGenerator.new("./")
    @parser = MPIParser.new("./test/fixtures/doc/xml/")
    @classes = %w{Wheel Tire Rim Car Label}
    @resolved_classes, @resolved_parents, @template_params, @class_sortation, @headers =
      @parser.shallow_resolution(@classes)
  end

  def test_generate_boost_serialize_function
    classes = %w{Car}
    members = @parser.shallow_resolution(classes)[0]

    actual_def = @generator.generate_serialize_function(classes[0],
                                                        members[classes[0]],
                                                        @resolved_parents[classes[0]],
                                                        @template_params[classes[0]])
    expected_def = File.read("./test/fixtures/references/boostserialize.h")

    assert_equal(expected_def, actual_def)
  end
end
