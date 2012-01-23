// z Koordinate absolut, x und y relativ
__kernel void execute(
    __global struct Cell* idata, __global struct Cell* odata,
    int zDim, int yDim, int xDim, int planes, int offsetZ,
    int offsetY, int offsetX, __global int* startCoords, __global int* endCoords,
    int actualX, int actualY)
{
    int indexX = get_global_id(0) + offsetX;
    int indexY = get_global_id(1) + offsetY;
    //int indexZ = get_global_id(2) * planes + offsetZ;
    
    int startStopIndex = get_global_id(2) * actualX * actualY + get_global_id(1) * actualX + get_global_id(0);
    int minZ = startCoords[startStopIndex];
    int maxZ = endCoords[startStopIndex];
    
    int index = minZ * xDim * yDim + indexY * xDim + indexX;
    
    for (; minZ < maxZ; ++minZ) {
        update(odata + index,
                idata + index - xDim * yDim,
                idata + index - xDim,
                idata + index,
                idata + index + xDim,
                idata + index + xDim * yDim);
        index += xDim * yDim;
    }
}
