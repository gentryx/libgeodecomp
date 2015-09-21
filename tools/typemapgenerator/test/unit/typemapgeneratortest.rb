require 'test/unit'
load 'typemapgenerator.rb'

class TestTypeParser < Test::Unit::TestCase
  def setup
    @mpi_generator = MPIGenerator.new("./")
    @boost_generator = BoostGenerator.new("./")
    @hpx_generator = HPXGenerator.new("./")

    @parser = MPIParser.new("./test/fixtures/doc/xml/", true)

    @mpi_classes = ["Car",
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

    @mpi_classes.sort!
    @boost_classes = @parser.find_classes_to_be_serialized("BoostSerialization").sort
    @hpx_classes = @parser.find_classes_to_be_serialized("HPXSerialization").sort

    @mpi_res = @parser.resolve_forest(@mpi_classes)
    @boost_res = @parser.shallow_resolution(@boost_classes)
    @hpx_res = @parser.shallow_resolution(@boost_classes)
  end

  # We use the real MPIParser and MPIGenerator since this way we get a
  # simple integration test for free.
  def test_generate_forest
    expected = @boost_generator.generate_forest(*@boost_res) +
      @hpx_generator.generate_forest(*@hpx_res) +
      @mpi_generator.generate_forest(*@mpi_res)

    actual = TypemapGenerator.generate_forest("./test/fixtures/doc/xml/",
                                              "./",
                                              true)

    assert_equal(expected.size, actual.size)
    assert_equal(expected[0], actual[0])
    assert_equal(expected[1], actual[1])
    assert_equal(expected[2], actual[2])
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
    actual = TypemapGenerator.find_classes_to_be_serialized("./test/fixtures/doc/xml/", "Typemaps")
    actual.map! { |header| header.gsub(/(.+\/)(fixtures.+)/) { |match| $2 } }
    assert_equal(expected.to_set, actual.to_set)
  end
end
