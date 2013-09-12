#include "libgeodecomp.cl"

__kernel void update(__constant coords_ctx * coords,
                     __constant void * in, __global void * out)
{
  double * i = (double *)in;
  double * o = (double *)out;

  o[get_address(coords, ( 0, 0, 0))] =
    ( i[get_address(coords, ( 0, 0,-1))]
    + i[get_address(coords, ( 0,-1, 0))]
    + i[get_address(coords, (-1, 0, 0))]
    + i[get_address(coords, ( 1, 0, 0))]
    + i[get_address(coords, ( 0, 1, 0))]
    + i[get_address(coords, ( 0, 0, 1))]
    ) * (1.0/6.0);
}
