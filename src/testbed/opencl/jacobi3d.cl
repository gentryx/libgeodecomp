double * value(double * v, uint2 x, uint2 y, uint2 z)
{
  return v + z.s0 * x.s1 * y.s1 + y.s0 * x.s1 + x.s0;
}

void update(int3 index, int3 size,
            __constant double * in, __global double * out)
{
  *value(out, (index.x, size.x), (index.y, size.y), (z, size.z) =
    ( *value(in, (index.x+1, size.x), (index.y+0, size.y), (index.z+0, size.z)
    + *value(in, (index.x+0, size.x), (index.y+1, size.y), (index.z+0, size.z)
    + *value(in, (index.x+0, size.x), (index.y+0, size.y), (index.z+1, size.z)
    + *value(in, (index.x+0, size.x), (index.y+0, size.y), (index.z+0, size.z)
    + *value(in, (index.x-1, size.x), (index.y+0, size.y), (index.z+0, size.z)
    + *value(in, (index.x+0, size.x), (index.y-1, size.y), (index.z+0, size.z)
    + *value(in, (index.x+0, size.x), (index.y+0, size.y), (index.z-1, size.z)
    ) / 7.0
    ;
}

__kernel void step(int3 size, __constant double * in, __global double * out)
{
  printf("x: %i, y: %i, z: %i\n", size.x, size.y, size.z);
  return;
  for (int x = 0; x < size.x; ++x)
    for (int y = 0; y < size.y; ++y)
      for (int z = 0; z < size.z; ++z)
        update((int3)(x, y, z), size);
}
