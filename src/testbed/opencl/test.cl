#include "libgeodecomp.cl"

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_intel_printf: enable

__kernel void stencil_test(__constant coords_ctx * coords,
                           __constant double * in, __global double * out)
{
  uint global_id = get_global_id(0);

  int3 point = coords->points[global_id];

  int index = (point.z - 1) * coords->points_size.x * coords->points_size.y
            + (point.y - 1) * coords->points_size.x
            + (point.x - 1);

  out[index] = 2 * in[index];
}

void
indices_test_pp(size_t gid, int xo, int yo, int zo, double v)
{
  printf("in[%u + ", gid);
  printf("(%i", xo);
  printf(", %i", yo);
  printf(", %i)] = ", zo);
  printf("%f\n", v);
}

__kernel void
indices_test(__constant coords_ctx * coords,
             __constant double * in, __global double * out)
{
  size_t gid = get_global_id(0);

  for (int i = 0; i < 1000 * gid * gid; ++i) {}

  int xo = 0, yo = 0, zo = 0;
  indices_test_pp(gid, xo, yo, zo, in[get_address(coords, (int3)(xo, yo, zo))]);

  xo = -1, yo = 0, zo = 0;
  indices_test_pp(gid, xo, yo, zo, in[get_address(coords, (int3)(xo, yo, zo))]);

  xo = 0, yo = -1, zo = 0;
  indices_test_pp(gid, xo, yo, zo, in[get_address(coords, (int3)(xo, yo, zo))]);

  xo = 0, yo = 0, zo = -1;
  indices_test_pp(gid, xo, yo, zo, in[get_address(coords, (int3)(xo, yo, zo))]);

  xo = -1, yo = -1, zo = -1;
  indices_test_pp(gid, xo, yo, zo, in[get_address(coords, (int3)(xo, yo, zo))]);

  xo = 1, yo = 0, zo = 0;
  indices_test_pp(gid, xo, yo, zo, in[get_address(coords, (int3)(xo, yo, zo))]);

  xo = 0, yo = 1, zo = 0;
  indices_test_pp(gid, xo, yo, zo, in[get_address(coords, (int3)(xo, yo, zo))]);

  xo = 0, yo = 0, zo = 1;
  indices_test_pp(gid, xo, yo, zo, in[get_address(coords, (int3)(xo, yo, zo))]);

  xo = 1, yo = 1, zo = 1;
  indices_test_pp(gid, xo, yo, zo, in[get_address(coords, (int3)(xo, yo, zo))]);
}
