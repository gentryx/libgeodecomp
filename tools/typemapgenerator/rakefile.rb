require 'pathname'
mydir = Pathname.new(pwd)

task :test do
  cd mydir
  FileList["test/**/*test.rb"].each do |test|
    sh "ruby #{test}"
  end
end
