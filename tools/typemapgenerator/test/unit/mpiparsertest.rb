require 'test/unit'
require 'timeout'
load 'mpiparser.rb'
load 'typemapgenerator.rb'

class TestMPIParser < Test::Unit::TestCase
  def setup
    @parser = MPIParser.new("./test/fixtures/doc/xml/")
  end

  def test_get_members_fom_car
    members = @parser.get_members("Car")
    expected = {
      "wheels" => {
        :type => "Wheel",
        :cardinality => "NumWheels"
      },
      "engine" => {
        :type => "Engine",
        :cardinality => 1
      }
    }

    assert_equal(expected, members)
  end

  def test_get_parents
    assert_equal(%w{Car Luxury}, @parser.get_parents("BMW"))
    assert_equal([],             @parser.get_parents("Car"))
  end

  def test_get_members_from_engine
    members = @parser.get_members("Engine")
    expected = {
      "capacity" => {
        :type => "double",
        :cardinality => 1
      },
      "gearRatios" => {
        :type => "double",
        :cardinality => 6
      },
      "fuel" => {
        :type => "Fuel",
        :cardinality => 1
      }

    }

    assert_equal(expected, members)
  end

  def test_get_members_from_floatcoord
    members = @parser.get_members("FloatCoord")
    expected = {
      "vec" => {
        :type => "float",
        :cardinality => "DIMENSIONS"
      }
    }

    assert_equal(expected, members)
  end

  def test_resolve_class_simple
    @parser = MPIParser.new("./test/fixtures/doc/xml/", true)

    klass = "Coord<2>"
    members = {
      "x" => {
        :type => "int",
        :cardinality => 1
      },
      "y" => {
        :type => "int",
        :cardinality => 1
      }
    }
    parents = []
    classes = %w(Coord<2>)
    resolved_classes = { }
    resolved_parents = { }
    topological_class_sortation = []
    is_abstract = {}

    @parser.resolve_class_simple(klass, members, parents, classes,
                                 resolved_classes,
                                 resolved_parents,
                                 topological_class_sortation,
                                 is_abstract)

    resolved_classes["Coord<2>"].each do |name, spec|
      assert_equal("MPI_INT", spec[:type])
    end
  end

  def test_resolve_class_success
    klass = "Rim"
    classes = %w(Apple Melon Rim)
    resolved_classes = { }
    resolved_parents = { }
    topological_class_sortation = []
    is_abstract = {}

    @parser.resolve_class(klass,
                          classes,
                          resolved_classes,
                          resolved_parents,
                          topological_class_sortation,
                          is_abstract)

    expected_typemap = Datatype.new
    expected_typemap.merge!(@parser.map_enums)
    expected_typemap["Rim"] = Datatype.cpp_to_mpi("Rim")

    members = @parser.get_members("Rim")

    members.values.each do |map|
      map[:class] = map[:type]
      map[:type] = expected_typemap[map[:type]]
    end
    expected_resolved_classes = { }
    expected_resolved_classes["Rim"] = members

    assert_equal(["Apple", "Melon"], classes)
    assert_equal(expected_typemap, @parser.datatype_map)
    assert_equal(expected_resolved_classes, resolved_classes)
    assert_equal(["Rim"], topological_class_sortation)
  end

  def test_resolve_class_failure
    klass = "Wheel"
    classes = %{Apple Melon Wheel}
    resolved_classes = { }
    resolved_parents = { }
    topological_class_sortation = []
    is_abstract = {}

    @parser.resolve_class(klass,
                          classes,
                          resolved_classes,
                          resolved_parents,
                          topological_class_sortation,
                          is_abstract)

    assert_equal(%{Apple Melon Wheel}, classes)
    assert_equal({ }, resolved_classes)
    assert_equal([], topological_class_sortation)
  end

  def test_template_parameters
    expected_params1 = [
      {
        :type => "typename",
        :name => "A"
      },
      {
        :type => "typename",
        :name => "B"
      }
    ]

    expected_params2 = [
      {
        :type => "int",
        :name => "DIMENSIONS"
      }
    ]

    assert_equal(expected_params1, @parser.template_parameters("CoordPair"))
    assert_equal([],               @parser.template_parameters("Car"))
    assert_equal(expected_params2, @parser.template_parameters("FloatCoord"))
  end

  def test_used_template_parameters1
    expected = [["Coord<3 >", "Coord<2 >"],
                ["int", "double"],
                ["int", "int"]]
    assert_equal(expected, @parser.used_template_parameters("CoordPair"))
  end

  def test_used_template_parameters2
    expected = [["2"], ["3"], ["4"]]
    assert_equal(expected.to_set,
                 @parser.used_template_parameters("CoordContainer").to_set)
  end

  def test_used_template_parameters3
    expected = [["1"], ["2"], ["4"]]
    assert_equal(expected.to_set,
                 @parser.used_template_parameters("FloatCoord").to_set)
  end

  def test_resolve_class_with_fixed_template_parameters
    klass = "Coord<2 >"
    classes = ["Coord<2 >", "Coord<3 >"]
    resolved_classes = { }
    resolved_parents = { }
    topological_class_sortation = []
    is_abstract = {}

    @parser.resolve_class("Coord<2 >",
                          classes,
                          resolved_classes,
                          resolved_parents,
                          topological_class_sortation,
                          is_abstract)

    @parser.resolve_class("Coord<3 >",
                          classes,
                          resolved_classes,
                          resolved_parents,
                          topological_class_sortation,
                          is_abstract)

    coord2_spec = {
      "x"=> { :type=>"MPI_INT", :cardinality=>1, :class => "int"},
      "y"=> { :type=>"MPI_INT", :cardinality=>1, :class => "int"}
    }
    coord3_spec = {
      "x"=> { :type=>"MPI_INT", :cardinality=>1, :class => "int"},
      "y"=> { :type=>"MPI_INT", :cardinality=>1, :class => "int"},
      "z"=> { :type=>"MPI_INT", :cardinality=>1, :class => "int"}
    }

    expected_resolved = {
      "Coord<2 >" => coord2_spec,
      "Coord<3 >" => coord3_spec
    }

    assert_equal(expected_resolved, resolved_classes)
  end

  def test_resolve_class_with_template_parameters_strict
    classes = ["Coord<2 >", "Coord<3 >", "CoordPair"]
    resolved_classes = { }
    resolved_parents = { }
    topological_class_sortation = []
    is_abstract = {}

    @parser.resolve_class("Coord<2 >",
                          classes,
                          resolved_classes,
                          resolved_parents,
                          topological_class_sortation,
                          is_abstract)

    @parser.resolve_class("Coord<3 >",
                          classes,
                          resolved_classes,
                          resolved_parents,
                          topological_class_sortation,
                          is_abstract)

    @parser.resolve_class("CoordPair",
                          classes,
                          resolved_classes,
                          resolved_parents,
                          topological_class_sortation,
                          is_abstract)

    @parser.resolve_class("CoordContainer",
                          classes,
                          resolved_classes,
                          resolved_parents,
                          topological_class_sortation,
                          is_abstract)

    expected1 = {
      "a" => { :type=>"MPI_COORD_3_", :cardinality=>1, :class => "Coord<3 >"},
      "b" => { :type=>"MPI_COORD_2_", :cardinality=>1, :class => "Coord<2 >"}
    }
    expected2 = {
      "a" => { :cardinality=>1, :type=>"MPI_INT",    :class => "int"},
      "b" => { :cardinality=>1, :type=>"MPI_DOUBLE", :class => "double"}
    }

    assert_equal(expected1,
                 resolved_classes["CoordPair<Coord<3 >,Coord<2 > >"])
    assert_equal(expected2,
                 resolved_classes["CoordPair<int,double >"])
  end

  def test_resolve_class_with_template_parameters_sloppy
    @parser = MPIParser.new("./test/fixtures/doc/xml/", true)
    @parser.type_hierarchy_closure.delete("Coord<1 >")

    classes = ["Coord<2 >", "Coord<3 >",
      "CoordContainer", "CoordContainerContainer"]
    resolved_classes = { }
    resolved_parents = { }
    topological_class_sortation = []
    is_abstract = {}

    @parser.resolve_class("Coord<2 >",
                          classes,
                          resolved_classes,
                          resolved_parents,
                          topological_class_sortation,
                          is_abstract)

    @parser.resolve_class("Coord<3 >",
                          classes,
                          resolved_classes,
                          resolved_parents,
                          topological_class_sortation,
                          is_abstract)

    @parser.resolve_class("CoordContainer",
                          classes,
                          resolved_classes,
                          resolved_parents,
                          topological_class_sortation,
                          is_abstract)

    @parser.resolve_class("CoordContainerContainer",
                          classes,
                          resolved_classes,
                          resolved_parents,
                          topological_class_sortation,
                          is_abstract)

    expected1 = nil
    expected2 = {
      "pos" => {
        :type => "MPI_COORD_2_",
        :cardinality => 1,
        :class => "Coord<2 >"
      }
    }
    expected3 = {
      "pos" => {
        :type => "MPI_COORD_3_",
        :cardinality => 1,
        :class => "Coord<3 >"
      }
    }
    expected4 = { }
    expected5 = {
      "cargo2" => {
        :type => "MPI_COORDCONTAINER_2_",
        :cardinality => 1,
        :class => "CoordContainer<2 >"
      },
      "cargo3" => {
        :type => "MPI_COORDCONTAINER_3_",
        :cardinality => 1,
        :class => "CoordContainer<3 >"
      },
      "cargo4" => {
        :type => "MPI_COORDCONTAINER_4__PARTIAL",
        :cardinality => 1,
        :class => "CoordContainer<4 >"
      }
    }

    assert_equal(expected1, resolved_classes["CoordContainer<1 >"])
    assert_equal(expected2, resolved_classes["CoordContainer<2 >"])
    assert_equal(expected3, resolved_classes["CoordContainer<3 >"])
    assert_equal(expected4, resolved_classes["CoordContainer<4 >"])
    assert_equal(expected5, resolved_classes["CoordContainerContainer"])
  end

  def test_find_enums
    assert_equal(["Fuel"], @parser.find_enums)
  end

  def test_resolve_forest_success
    classes = %w{Engine Car Wheel Rim Tire}.to_set
    opts = @parser.resolve_forest(classes)

    opts.headers.map! { |elem| elem.gsub!(/(.+\/)(fixtures.+)/) { |match| $2 } }

    expected_datatype_map = Datatype.new
    expected_datatype_map.merge!(@parser.map_enums)
    classes.each do |klass|
      expected_datatype_map[klass] = Datatype.cpp_to_mpi(klass)
    end

    car_map = {
      "engine" => {
        :class => "Engine",
        :type => expected_datatype_map["Engine"],
        :cardinality => 1
      },
      "wheels" => {
        :class => "Wheel",
        :type => expected_datatype_map["Wheel"],
        :cardinality => "NumWheels"
      }
    }

    engine_map = {
      "capacity" => {
        :class => "double",
        :type => expected_datatype_map["double"],
        :cardinality => 1
      },
      "fuel" => {
        :class => "Fuel",
        :type => expected_datatype_map["int"],
        :cardinality => 1
      },
      "gearRatios" => {
        :class => "double",
        :type => expected_datatype_map["double"],
        :cardinality => 6
      }
    }

    wheel_map = {
      "rim" => {
        :class => "Rim",
        :type => expected_datatype_map["Rim"],
        :cardinality => 1
      },
      "tire" => {
        :class => "Tire",
        :type => expected_datatype_map["Tire"],
        :cardinality => 1
      }
    }

    rim_map = {
      "chromePlated" => {
        :class => "bool",
        :type => expected_datatype_map["bool"],
        :cardinality => 1
      }
    }

    tire_map = {
      "treadDepth" => {
        :class => "double",
        :type => expected_datatype_map["double"],
        :cardinality => 1
      }
    }

    expected_classes = {
      "Car" => car_map,
      "Engine" => engine_map,
      "Wheel" => wheel_map,
      "Rim" => rim_map,
      "Tire" => tire_map
    }

    expected_sortation = %w{Engine Rim Tire Wheel Car}

    expected_headers = [
      "fixtures/src/engine.h",
      "fixtures/src/rim.h",
      "fixtures/src/tire.h",
      "fixtures/src/wheel.h",
      "fixtures/src/car.h"]

    assert_equal(expected_datatype_map, opts.datatype_map)
    assert_equal(expected_classes,      opts.resolved_classes)
    assert_equal(expected_sortation,    opts.topological_class_sortation)
    assert_equal(expected_headers,      opts.headers)
  end

  def test_resolve_forest_failure
    classes = ["Wheel"]

    # catch inf. loops
    Timeout::timeout(10) do
      assert_raise(RuntimeError) { @parser.resolve_forest(classes) }
    end
  end

  def test_incomplete_strict_parsing_should_fail
    assert_raise(RuntimeError) do
      @parser.resolve_forest(%w(CarContainer))
    end
  end

  def test_find_classes_to_be_serialized1
    expected_classes = ["Coord<2 >", "Coord<3 >"] +
      %w{CoordContainer FloatCoord FloatCoordTypemapsHelper CoordContainerContainer CoordPair Dummy Engine BMW Car Wheel Rim Tire CarContainer Luxury}
    expected_classes = expected_classes.to_set

    assert_equal(expected_classes, @parser.find_classes_to_be_serialized("Typemaps"))
  end

  def test_find_classes_to_be_serialized2
    expected_classes = ["Coord<1 >", "Coord<3 >"] +
      %w{Engine Car Wheel Rim Tire Label}
    expected_classes = expected_classes.to_set
    actual_classes = @parser.find_classes_to_be_serialized("BoostSerialization")

    assert_equal(expected_classes, actual_classes)
  end
end


class TestSloppyParsing < Test::Unit::TestCase
  def setup
    @parser = MPIParser.new("./test/fixtures/doc/xml/", true)
  end

  def test_incomplete_sloppy_parsing_should_succeed
    assert_nothing_raised do
      parser = MPIParser.new("./test/fixtures/doc/xml/", true)
      parser.resolve_forest(%w(Engine Car Wheel Rim Tire CarContainer))
    end
  end

  def test_incomplete_sloppy_parsing_should_yield_correct_members
    @parser.type_hierarchy_closure.delete "Engine"
    classes = %w(Car Wheel Rim Tire CarContainer)
    resolved_classes = @parser.resolve_forest(classes).resolved_classes

    assert_equal(%w(wheels), resolved_classes["Car"].keys)
    assert_equal(%w(size spareWheel), resolved_classes["CarContainer"].keys.sort)
  end

  def test_incomplete_sloppy_parsing_should_yield_correct_typemap_names
    datatype_map = @parser.resolve_forest(%w(CarContainer Car Engine Wheel Rim Tire)).datatype_map

    %w(Car Engine Wheel Rim Tire).each do |klass|
      assert_equal(Datatype.cpp_to_mpi(klass), datatype_map[klass])
    end

    assert_equal(Datatype.cpp_to_mpi("CarContainer", true),
                 datatype_map["CarContainer"])
  end
end

class TestMapMemberTypesToMPI_Datatypes < Test::Unit::TestCase
  def setup
    @parser = MPIParser.new("./test/fixtures/doc/xml/")

    @members = {
      "x" => {
        :cardinality => 1,
        :type => "int"
      },
      "y" => {
        :cardinality => 1,
        :type => "double"
      }
    }

    @expected = {
      "x" => {
        :cardinality => 1,
        :type => "MPI_INT",
        :class => "int"
      },
      "y" => {
        :cardinality => 1,
        :type => "MPI_DOUBLE",
        :class => "double"
      }
    }
  end

  def test_builtin_types_should_succeed
    assert_equal(@expected,
                 @parser.map_types_to_MPI_Datatypes(@members))
  end

  def test_userdefined_type_should_fail
    @members = {
      "x" => {
        :cardinality => 1,
        :type => "int"
      },
      "a" => {
        :cardinality => 1,
        :type => "Gobble"
      }
    }

    assert_raise(RuntimeError) do
      @parser.map_types_to_MPI_Datatypes(@members)
    end
  end

  def test_declared_userdefined_type_should_succeed
    @members = {
      "x" => {
        :cardinality => 1,
        :type => "int",
      },
      "a" => {
        :cardinality => 1,
        :type => "Gobble"
      }
    }

    @expected = {
      "x" => {
        :cardinality => 1,
        :type => "MPI_INT",
        :class => "int"
      },
      "a" => {
        :cardinality => 1,
        :type => "MPI_GOBBLE",
        :class => "Gobble"
      }
    }

    @parser.datatype_map["Gobble"] = "MPI_GOBBLE"
    assert_equal(@expected,
                 @parser.map_types_to_MPI_Datatypes(@members))
  end

  def test_find_header
    actual = @parser.find_header("Car").gsub(/(.+\/)(fixtures.+)/) { |match| $2 }
    assert_equal("fixtures/src/car.h", actual)
  end
end
