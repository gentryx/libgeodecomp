#include "libgeodecomp.cl"

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

__kernel void coords_test(__constant coords_ctx * coords,
                          __constant double * in, __global double * out)
{
  int index = coords->indices[get_global_id(0)];
  out[index] += in[index];
  printf("{%u, (%v3d)} ", get_global_id(0), coords->points[get_global_id(0)]);
  printf("in[%d] = %f\n", index, in[index]);
}

__kernel void test(int3 base, __constant int3 * points,
                   __constant double * in, __global double * out)
{
  uint work_dim = get_work_dim();
  size_t global_id = get_global_id(0);
  printf("work_dim: %u, ", work_dim);
  printf("global_id %u, ", global_id);
  printf("point %v3d, ", points[global_id]);
  int x = points[global_id].x - 1;
  int y = points[global_id].y - 1;
  int z = points[global_id].z - 1;
  printf(": %f, ", in[z * base.x * base.y + y * base.x + x]);
  printf("\n");
}
