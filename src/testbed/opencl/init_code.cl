typedef struct {
           int4   points_size;
  __global int4 * points;
  __global int  * indices;
} coords_ctx;

__kernel void
mem_hook(__global void * coords_in,
         __global int4 * points,
         __global int * indices)
{
  __global coords_ctx * coords = (__global coords_ctx*)coords_in;
  coords->points = points;
  coords->indices = indices;
}

__kernel void
data_init(__constant void * coords_in)
{
  __constant coords_ctx * coords = (__constant coords_ctx*)coords_in;

  size_t gid = get_global_id(0);
  int4 size = coords->points_size;
  size_t x = coords->points[gid].s0;
  size_t y = coords->points[gid].s1;
  size_t z = coords->points[gid].s2;

  coords->indices[gid] = ((z + size.z) % size.z) * size.x * size.y
                       + ((y + size.y) % size.y) * size.x
                       + ((x + size.x) % size.x);
}
