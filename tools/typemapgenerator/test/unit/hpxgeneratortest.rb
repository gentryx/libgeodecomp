require 'test/unit'
load 'hpxgenerator.rb'
load 'mpiparser.rb'

class TestHPXGenerator < Test::Unit::TestCase
  def setup
    @generator = HPXGenerator.new("./")
    @parser = MPIParser.new("./test/fixtures/doc/xml/")
    @classes = %w{Wheel Tire Rim Car Label}
    @resolved_classes, @resolved_parents, @template_params, @class_sortation, @headers =
      @parser.shallow_resolution(@classes)
  end

  def test_generate_hpx_serialize_function
    classes = %w{Car}
    members = @parser.shallow_resolution(classes)[0]

    actual_def = @generator.generate_serialize_function(classes[0],
                                                        members[classes[0]],
                                                        @resolved_parents[classes[0]],
                                                        @template_params[classes[0]])
    expected_def = File.read("./test/fixtures/references/hpxserialize.h")

    assert_equal(expected_def, actual_def)
  end
end
