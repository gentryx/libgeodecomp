require 'test/unit'
load 'datatype.rb'

class DatatypeTest < Test::Unit::TestCase
  def setup
    @datatype = Datatype.new
  end

  # case 1: C++ name is passed verbosely
  def test_char
    assert_equal('MPI_CHAR', @datatype['char'])
    assert_equal('MPI_UNSIGNED_CHAR', @datatype['unsigned char'])
    assert_equal('MPI_SIGNED_CHAR', @datatype['signed char'])
  end

  # case 2: special signed/unsigned handling
  def test_int
    assert_equal('MPI_INT', @datatype['int'])
  end

  # case 3: signed/unsigned (see 2) and optional 'int' suffix
  def test_long
    assert_equal('MPI_LONG', @datatype['long'])
    assert_equal('MPI_UNSIGNED_LONG', @datatype['unsigned long'])
  end

  def test_lookup_unknown
    assert_nil @datatype['foo']
  end

  def test_nameconversion_unknown
    assert_equal('MPI_FOOBAR', Datatype.cpp_to_mpi('FooBar'))
  end

  def test_map
    assert_operator(@datatype.size, :>=, 14)
    @datatype.each_pair do |key, value|
      assert_equal(value, @datatype[key])
    end
  end
end
