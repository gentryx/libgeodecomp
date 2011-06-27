require 'test/common'

class RakeTest < Test::Unit::TestCase
  include UtilityFunctions

  @@default_src = "src_crossplatform"
  @@conf_args = "--disable-typemaps"

  def test_crossplatform
    log = build("installer")
    assert(/CPack: Package (.*testpackage-r12345-.*\.exe) generated/ =~ log)
    assert(File.exist?($1))
  end  
end
