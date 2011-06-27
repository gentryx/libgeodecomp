#!/bin/bash
set -e
doxygen doxygen.conf
../generate.rb -S doc/xml src
cd src
make
./demo
