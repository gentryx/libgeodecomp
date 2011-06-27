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
require 'datatype'

class DatatypeTest < Test::Unit::TestCase
  def setup
    @datatype = Datatype.new
  end

  # case 1: C++ name is passed verbosely
  def test_char
    assert_equal('MPI::CHAR', @datatype['char'])
    assert_equal('MPI::UNSIGNED_CHAR', @datatype['unsigned char'])
    assert_equal('MPI::SIGNED_CHAR', @datatype['signed char'])
  end

  # case 2: special signed/unsigned handling
  def test_int
    assert_equal('MPI::INT', @datatype['int'])
  end

  # case 3: signed/unsigned (see 2) and optional 'int' suffix
  def test_long
    assert_equal('MPI::LONG', @datatype['long'])
    assert_equal('MPI::UNSIGNED_LONG', @datatype['unsigned long'])
  end
  
  def test_lookup_unknown
    assert_nil @datatype['foo']
  end

  def test_nameconversion_unknown
    assert_equal('MPI::FOOBAR', Datatype.cpp_to_mpi('FooBar'))
  end

  def test_map
    assert_operator(@datatype.size, :>=, 14)
    @datatype.each_pair do |key, value|
      assert_equal(value, @datatype[key])
    end
  end
end
