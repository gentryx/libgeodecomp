typedef struct {
  int    num_points;
  int    neighbors_per_point;
  int3   points_size;
  __constant int3 * points;
  __constant int  * indices;
} coords_ctx;

