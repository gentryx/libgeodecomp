#ifndef LIBGEODECOMP_STORAGE_SELL_C_SIGMA_SPARSEMATRIX_CONTAINER_H
#define LIBGEODECOMP_STORAGE_SELL_C_SIGMA_SPARSEMATRIX_CONTAINER_H

#include <vector>
#include <utility>
#include <assert.h>
#include <stdexcept>

#include <iostream>

namespace LibGeoDecomp {


//OHNE SORTIEREN! SIGMA =1 TODO
template<typename VALUETYPE, int C = 1, int SIGMA = 1>
class SellCSigmaSparseMatrixContainer
{
public:
    SellCSigmaSparseMatrixContainer(){
        if (SIGMA != 1 && SIGMA % C != 0){
            throw std::invalid_argument("SIGMA must be '1' or multiple of C");
        }
    }

    void matVecMul (std::vector<double> lhs, std::vector<double> rhs){
        //TODO
    }

    std::vector< std::pair<int, VALUETYPE> > getRow(int const row){

        std::vector< std::pair<int, VALUETYPE> > vec;
        int const chunk (row/C);
        int const offset (row%C);
        int index = chunkOffset[chunk] + offset;

//std::cout << "Get row: row="<< row << " chunk=" << chunk << " offset=" << offset << " chunkOffset=" << chunkOffset[chunk] << " Werte=";

        for (int element = 0;
                element < rowLength[row]; ++element, index += C){
            vec.push_back( std::pair<int, VALUETYPE>
                            (column[index], values[index]) );
//std::cout << "(" << vec.back().first << ","<< vec.back().second << "), ";
        }

//std::cout << std::endl;
        return vec;
    }

    // Row [0:N-1]; Col [0:N-1]
    void addPoint(int const row, int const col, VALUETYPE value){
        if(row < 0 && col < 0){
            throw std::invalid_argument("row and colum must be >= 0");
        }

        int const chunk (row/C);

//std::cout << "Add point: row: " << row << " chunk: " << chunk << " col=" << col << " val=" << value;

        if ( (unsigned)row >= rowLength.size() ){
            rowLength.resize(row+1);
        }
        if ( (unsigned)chunk >= chunkLength.size() ){
            unsigned oldNumberOfChunks = chunkLength.size();

            chunkLength.resize(chunk+1);
            chunkOffset.resize(chunk+2);

            for( unsigned i = oldNumberOfChunks; i < chunkOffset.size(); ++i ){
                chunkOffset[i] = values.size();
            }
        }

//std::cout << " row length: " << rowLength[row] << " chunk Length: " << chunkLength[chunk] << " chunkOffset: " << chunkOffset[chunk] << std::endl;

        //// case 1: row is NOT the bigest in chunk
        if ( rowLength[row] < chunkLength[chunk] ){
//std::cout << "case 1";
            std::vector<int>::iterator itCol = column.begin()
                    + chunkOffset[chunk] + row % C;

            while ( col > *itCol && -1 != *itCol ){
                itCol += C;
            }
            if(col == *itCol){
                *itCol = col;
                values[itCol - column.begin()] = value;
                return;
            }
            
            if ( -1 != *itCol){
            //// case 1.a add value in mid of row
//std::cout << ".a";
                int lastElement = chunkOffset[chunk + 1] - C + (row%C);
                int end   = itCol - column.begin();

                for (int i = lastElement; i > end; i-=C){
                    values[i] = values[i-C];
                    column[i] = column[i-C];
                }
            }
//std::cout << std::endl;

            values[itCol - column.begin()] = value;
            *itCol = col;

            ++rowLength[row];
        }
        else{
        //// case 2: row is the logest in chunk -> expend chunk
//std::cout << "fall 2";

            int const offset    = chunkOffset[chunk] + row % C;
            int const offsetEnd = chunkOffset[chunk+1];

            int index = offset;
            while (index < offsetEnd && col > column[index]){
                index += C;
            }

            if (index >= offsetEnd ){
//std::cout << ".a" << std::endl;
                index = offsetEnd;

                std::vector<int>::iterator itCol = column.begin() + index;
                typename
                std::vector<VALUETYPE>::iterator itVal = values.begin() + index;

                for (int i=0; i < C; ++i){
                        itCol = column.insert(itCol, -1);               
                        itVal = values.insert(itVal, VALUETYPE());
                }
                *(itCol + (row%C)) = col;
                *(itVal + (row%C)) = value;
            }
            else {
//std::cout << ".b" << std::endl;
                if(col == column[index]){
                    column[index] = col;
                    values[index] = value;
                    return;
                }

                std::vector<int>::iterator itCol = column.begin() + index;
                typename
                std::vector<VALUETYPE>::iterator itVal = values.begin() + index;

                for (int i=0; i < C; ++i){
                    itCol = column.insert(itCol, -1);               
                    itVal = values.insert(itVal, VALUETYPE());
                }
                *itVal = value;
                *itCol = col;

                //// fix order
                for ( int i = index; i < offsetEnd; ++i ){
                    if (i%C != row%C){
                        values[i] = values[i + C];
                        column[i] = column[i + C];
                    }
                }
                for (int i = 0; i < C; ++i ){
                    if(i != row % C){
                        values[offsetEnd + i] = VALUETYPE();
                        column[offsetEnd + i] = -1;
                    }
                }
            }

            ++rowLength[row];
            chunkLength[chunk] = rowLength[row];
            for (unsigned ch = chunk+1; ch < chunkOffset.size(); ++ch){
                chunkOffset[ch] += C;
            }
        }

//std::cout << "col: ";
//for (unsigned i=0; i<column.size(); ++i){
    //std::cout << column[i] << " ";
//}
//std::cout << std::endl;
//std::cout << "values: ";
//for (unsigned i=0; i<values.size(); ++i){
    //std::cout << values[i] << " ";
//}
//std::cout << std::endl;

}

private:
    std::vector<VALUETYPE> values;
    std::vector<int>       column;
    std::vector<int>       rowLength;   // = Non Zero Entres in Row
    std::vector<int>       chunkLength; // = Max rowLength in Chunk
    std::vector<int>       chunkOffset; // COffset[i+1]=COffset[i]+CLength[i]*C
};

}

#endif
