require 'test/unit'
require 'boostgenerator'
require 'mpiparser'

class TestMPIGenerator < Test::Unit::TestCase
  def setup
    @generator = BoostGenerator.new("./")
    @parser = MPIParser.new("./test/fixtures/doc/xml/")
    @classes = %w{Wheel Tire Rim Car Label}
    @resolved_classes, @resolved_parents, @template_params, @class_sortation, @headers =
      @parser.shallow_resolution(@classes)
  end

  def test_basic
    # fixme
  end

end
