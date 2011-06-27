require 'test/common'

class RakeTest < Test::Unit::TestCase
  include UtilityFunctions

  @@default_src = "src_crossplatform"
  @@conf_args = "--no-typemaps"

  def test_crossplatform
    # output = sh(@opts.builddir + "testexe/testexe")
    # assert_match(/This is Hell World/, output)
    # assert_match(/ping/, output)
  end  
end
