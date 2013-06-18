typedef struct {
  int3   points_size;
  __constant int3 * points;
  __global int  * indices;
} coords_ctx;

size_t
get_address(__constant coords_ctx * coords, int3 offset)
{
  size_t gid = get_global_id(0);
  int3 size = coords->points_size;
  size_t x = coords->points[gid].s0 + offset.x;
  size_t y = coords->points[gid].s1 + offset.y;
  size_t z = coords->points[gid].s2 + offset.z;

  if (0 == (offset.x | offset.y | offset.z)) {
    return coords->indices[gid];
  } else {
    return ((z + size.z) % size.z) * size.x * size.y
         + ((y + size.y) % size.y) * size.x
         + ((x + size.x) % size.x);
  }
}
