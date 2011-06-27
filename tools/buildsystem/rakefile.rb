task :test => :test_crossplatform do
  system "ruby test/acceptance/raketest.rb"
end

task :test_windows => :test_crossplatform do
  system "ruby test/windows/raketest.rb"
end

task :test_crossplatform do
  system "ruby test/crossplatform/raketest.rb"
end
