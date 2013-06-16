__kernel void
mem_hook_up(__global coords_ctx * coords,
            __constant int3 * points, __constant int * indices)
{
  coords->points = points;
  coords->indices = indices;
}

__kernel void
compute_boundaries(__global coords_ctx * coords)
{
  uint global_id = get_global_id(0);
  int3 size = coords->points_size;
  int x = coords->points[global_id].x;
  int y = coords->points[global_id].y;
  int z = coords->points[global_id].z;

  coords->indices[global_id] = ((z + size.z) % size.z) * size.x * size.y
                             + ((y + size.y) % size.y) * size.x
                             + ((x + size.x) % size.x);
}
