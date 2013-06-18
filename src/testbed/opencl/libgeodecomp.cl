typedef struct {
  int    num_points;
  int    num_neighbors;
  int3   points_size;
  __constant int3 * points;
  __constant int  * indices;
} coords_ctx;

