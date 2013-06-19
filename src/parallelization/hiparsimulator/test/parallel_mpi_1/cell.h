#ifdef __cplusplus
// this makes it easier in our unit tests to locate this file
std::string cellSourceFile = __FILE__;

#ifndef __global
#define __global
#endif

#endif


struct Cell
{
  double val;
};

void update(__global struct Cell *target,
            __global struct Cell *back,
            __global struct Cell *up,
            __global struct Cell *same,
            __global struct Cell *down,
            __global struct Cell *front)
{
    target->val = same->val;
}
