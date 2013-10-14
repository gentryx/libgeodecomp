# Mapping of C++ datatypes names to MPI datatypes names. Support for
# built-in types as found in section 10.1.6 of MPI-2, see
# http://www.mpi-forum.org/docs/mpi-20-html/node229.htm. We also
# include the optional 'long long' types.
class Datatype < Hash
  def initialize
    [
      "char",
      "signed char",
      "unsigned char",
      "short",
      "unsigned short",
      "int",
      "unsigned",
      "long",
      # excluding "unsigned long" to avoid clashes with "size_t"
      # "unsigned long",
      "float",
      "double",
      "long double",
      "long long",
      "unsigned long long",
    ].each do |t|
      self[t] = Datatype.cpp_to_mpi(t)
    end

    self["bool"] = "MPI_CHAR"
    self["wchar_t"] = "MPI_WCHAR"
    self["std::complex<float>"] = "MPI_COMPLEX"
    self["std::complex<double>"] = "MPI_DOUBLE_COMPLEX"
    self["size_t"] = "MPI_UNSIGNED_LONG"
    # self["std::complex<long double>"] = "MPI_LONG_DOUBLE_COMPLEX"
  end

  class << self
    # Convert a C++ class or type name into the MPI equivalent: use
    # the built-in MPI name for primitive types, and generate a MPI
    # name for user-defined types. This is just a means for name
    # conversions, don't mix it up with the look-up operator []. The
    # loop-up operator must not in any case return some value other
    # than nil as this would prevent the MPIParser from recognizing
    # pending classes. If in doubt take a look at the calls to
    # Datatype#[] and Datatype::class2mpi.
    def cpp_to_mpi(type, partial = false)
      ret = "MPI_" + type.gsub(/[, :<>]+/, '_').upcase
      ret += "_PARTIAL" if partial
      return ret
    end
  end
end
