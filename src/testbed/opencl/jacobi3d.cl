#pragma OPENCL EXTENSION cl_khr_fp64: enable

typedef struct {
           int4   points_size;
  __global int4 * points;
  __global int  * indices;
} coords_ctx;

size_t
get_address(__constant coords_ctx * coords, int4 offset)
{
  size_t gid = get_global_id(0);

  if (0 == (offset.x | offset.y | offset.z)) {
    return coords->indices[gid];

  } else {
    int4 size = coords->points_size;
    size_t x = coords->points[gid].s0 + offset.x;
    size_t y = coords->points[gid].s1 + offset.y;
    size_t z = coords->points[gid].s2 + offset.z;

    return ((z + size.z) % size.z) * size.x * size.y
         + ((y + size.y) % size.y) * size.x
         + ((x + size.x) % size.x);
  }
}

__kernel void update(__constant void * coords,
                     __constant void * in, __global void * out)
{
  __constant double * i = (__constant double *)in;
  __global   double * o = (  __global double *)out;

  o[get_address(coords, ( 0, 0, 0))] =
    ( i[get_address(coords, ( 0, 0,-1))]
    + i[get_address(coords, ( 0,-1, 0))]
    + i[get_address(coords, (-1, 0, 0))]
    + i[get_address(coords, ( 1, 0, 0))]
    + i[get_address(coords, ( 0, 1, 0))]
    + i[get_address(coords, ( 0, 0, 1))]
    ) * (1.0/6.0);
}
