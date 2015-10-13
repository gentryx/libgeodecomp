#!/bin/bash

wget -c http://www.cise.ufl.edu/research/sparse/MM/Fluorem/RM07R.tar.gz
wget -c http://www.cise.ufl.edu/research/sparse/MM/Zaoui/kkt_power.tar.gz
wget -c http://www.cise.ufl.edu/research/sparse/MM/Janna/ML_Geer.tar.gz
wget -c http://www.cise.ufl.edu/research/sparse/MM/Hamrle/Hamrle3.tar.gz

for i in RM07R.tar.gz kkt_power.tar.gz ML_Geer.tar.gz Hamrle3.tar.gz; do
    tar -xf $i
    rm -f $i
done

for i in RM07R kkt_power ML_Geer Hamrle3; do
    mv $i/* .
    rm -rf $i
done
