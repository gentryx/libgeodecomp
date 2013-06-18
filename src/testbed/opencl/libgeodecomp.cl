typedef struct {
  int3   points_size;
  __constant int3 * points;
  __global int  * indices;
} coords_ctx;

