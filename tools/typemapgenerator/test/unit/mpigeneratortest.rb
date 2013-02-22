# Copyright (C) 2006,2007 Andreas Schaefer <gentryx@gmx.de>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301 USA.

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
                                               @headers)
    assert_match(/^namespace FooBar \{/, actual_source)
    assert_match(/^namespace FooBar \{/, actual_header)
  end
end
