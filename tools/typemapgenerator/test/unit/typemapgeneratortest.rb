require 'test/unit'
require 'typemapgenerator'

class TestTypeParser < Test::Unit::TestCase
  def setup
    @generator = MPIGenerator.new("./")
    @parser = MPIParser.new("./test/fixtures/doc/xml/", true)
    @classes = ["Car",
                "Coord<2 >",
                "Coord<3 >",
                "CoordPair",
                "CoordContainer",
                "CoordContainerContainer",
                "Dummy",
                "FloatCoord",
                "FloatCoordTypemapsHelper",
                "Rim",
                "Engine",
                "Tire",
                "BMW",
                "Wheel",
                "Luxury",
                "CarContainer"]
    @classes.sort!
    @res = @parser.resolve_forest(@classes)
  end
  
  # We use the real MPIParser and MPIGenerator since this way we get a
  # simple integration test for free.
  def test_generate_forest
    expected = @generator.generate_forest(*@res)
    actual = TypemapGenerator.generate_forest("./test/fixtures/doc/xml/",
                                              "./",
                                              true)
    assert_equal(expected, actual)
  end

  def test_find_classes_to_be_serialized
    expected = 
      ["fixtures/src/coordcontainercontainer.h",
       "fixtures/src/coordcontainer.h",
       "fixtures/src/coord.h",
       "fixtures/src/floatcoordbase.h",
       "fixtures/src/pair.h",
       "fixtures/src/rim.h",
       "fixtures/src/carcontainer.h",
       "fixtures/src/engine.h",
       "fixtures/src/bmw.h",
       "fixtures/src/luxury.h",
       "fixtures/src/wheel.h",
       "fixtures/src/tire.h",
       "fixtures/src/car.h"]
    actual = TypemapGenerator.find_classes_to_be_serialized("./test/fixtures/doc/xml/")
    actual.map! { |header| header.gsub(/(.+\/)(fixtures.+)/) { |match| $2 } }
    assert_equal(expected.to_set, actual.to_set)    
  end
end
