#include "interfaces.h"

CellVirtual::CellVirtual(double v) : val(v)
{}

void CellVirtual::update(HoodOld& hood)
{
    Coord<2> c1( 0, -1);
    Coord<2> c2(-1,  0);
    Coord<2> c3( 0,  0);
    Coord<2> c4( 1,  0);
    Coord<2> c5( 0,  1);
    val = 
        (hood[c1].getVal() + 
         hood[c2].getVal() +
         hood[c3].getVal() +
         hood[c4].getVal() +
         hood[c5].getVal()) * (1.0 / 5.0);
}

double CellVirtual::getVal()
{
    return val;
}
