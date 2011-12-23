require 'fileutils'
require 'tempfile'
require 'test/unit'
require 'ostruct'
require 'pathname'
require 'yaml'

module UtilityFunctions
  def setup(default_src="src", conf_args="", tempdir_name="raketest")
    @testdir = Pathname.new(__FILE__).parent
    @tmp_dir = gen_tempdir(tempdir_name)
    @install_dir = gen_tempdir(tempdir_name)
    workdir = @tmp_dir + "src"
    FileUtils.mkdir(workdir)

    FileUtils.mkdir(@tmp_dir + "tools")
    FileUtils.cp_r(@testdir + "fixtures" + "tools" + "typemapgenerator", @tmp_dir + "tools")
    FileUtils.cp_r(@testdir + "fixtures" + "lib", @tmp_dir)

    files = %w{rakefile.rb doxygen.conf configure.example}.map do |f|
      @testdir + ".." + "src" + f
    end
    FileUtils.cp_r(files, workdir)

    patch(default_src)
    
    @file_list = file_list
    configure(workdir, conf_args)
    File.open(workdir + "conf.cmake", "a") do |f|
      f.puts "set(CMAKE_INSTALL_PREFIX #@install_dir)"
    end
    @opts = File.open(workdir + "conf.yaml") { |f| YAML.load(f) }
    @log = build
  end

  def teardown
    FileUtils.rm_rf(@tmp_dir)
    FileUtils.rm_rf(@install_dir)
  end

  def file_list
    Dir.glob(@tmp_dir + "**" + "*")
  end
  
  def patch(src_dir)    
    # sleep 1 second to make make aware of updated timestamps (sadly
    # make/rake doesn't sport split second accuracy)
    sleep 1
    bogus_svn_dirs = Dir.glob(@tmp_dir + "**/.svn")
    FileUtils.rm_rf bogus_svn_dirs
    path = @testdir + "fixtures" + src_dir + "*"
    files = Dir.glob(path)
    assert(!files.empty?)
    target_src  = @tmp_dir + "src"
    FileUtils.cp_r(files, target_src)
    
    target_link = target_src + "packagefoo" 
    if !File.exist?(target_link)
      FileUtils.ln_s(target_src, target_link);
    end
  end
  
  def configure(path, args="")
    sh "cd '#{path}' && ./configure.example #{args}"
  end
  
  def build(target="test", path=@opts.srcdir, assert_success=true)
    # sleep 1 second to make make aware of updated timestamps (sadly
    # make/rake doesn't sport split second accuracy)
    sleep 1
    sh("cd '#{path}' && rake #{target}", assert_success)
  end
  
  def gen_tempdir(tempdir_name)
    file = Tempfile.new(tempdir_name)
    dir = file.path
    file.close!
    FileUtils.mkdir(dir)    
    return Pathname.new(dir)
  end
  
  def sh(command, assert_success=true)
    res = `#{command} 2>&1`
    if assert_success
      unless $?.success?
        puts "> #{command}"
        puts res 
      end
      assert_block "command failed" do
        $?.success?
      end
    end
    return res
  end
  
  def count_matching_lines(regex, log)
    matches = log.split("\n").find_all { |l| l =~ regex }
    matches.size
  end
  
  def assert_file_exists(file)
    assert_block "File #{file} doesn't exist" do
      File.exists?(file)
    end
  end
  
  def assert_file_doesnt_exist(file)
    assert_block "File #{file} exists when it shouldn't" do
      !File.exists?(file)
    end
  end
end
