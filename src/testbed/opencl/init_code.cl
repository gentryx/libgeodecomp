#include "libgeodecomp.cl"


__kernel void
mem_hook(__global coords_ctx * coords,
         __global int4 * points,
         __global int * indices)
{
  coords->points = points;
  coords->indices = indices;
}

__kernel void
data_init(__constant coords_ctx * coords)
{
  size_t gid = get_global_id(0);
  int4 size = coords->points_size;
  size_t x = coords->points[gid].s0;
  size_t y = coords->points[gid].s1;
  size_t z = coords->points[gid].s2;

  coords->indices[gid] = ((z + size.z) % size.z) * size.x * size.y
                       + ((y + size.y) % size.y) * size.x
                       + ((x + size.x) % size.x);
}
