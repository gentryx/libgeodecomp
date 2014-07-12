#ifndef LIBGEODECOMP_STORAGE_SELLCSIGMASPARSEMATRIXCONTAINER_H
#define LIBGEODECOMP_STORAGE_SELLCSIGMASPARSEMATRIXCONTAINER_H

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
    explicit SellCSigmaSparseMatrixContainer(){}

    SellCSigmaSparseMatrixContainer(int N):
       values(), column(), rowLength(N, 0),
       chunkLength((N-1)/C + 1, 0), chunkOffset((N-1)/C + 2, 0), dimension(N)
    {
        if (C < 1 || SIGMA != 1 ){
            throw std::invalid_argument("SIGMA must be '1'; everithing else is not implemented jet");
        }
    }

    // lhs = A   x rhs
    // tmp = val x b
    void matVecMul (std::vector<VALUETYPE> & lhs, std::vector<VALUETYPE> & rhs){
        if(lhs.size() != rhs.size() || lhs.size() != dimension){
            throw std::invalid_argument("lhs and rhs must be of size N");
        }

        // loop over chunks     TODO paralel omp
        for(size_t chunk=0; chunk<chunkLength.size(); ++chunk){
            int offs = chunkOffset[chunk];
            VALUETYPE tmp[C];

            // init tmp                     TODO vectorise
            for(int row=0; row<C; ++row){
                tmp[row] = lhs[chunk*C + row];
            }

            // loop over columns in chunk
            for(int col=0; col<chunkLength[chunk]; ++col){

                // loop over rows in chunks TODO vectorise
                for(int row=0; row<C; ++row){
                    VALUETYPE val = values[offs];
                    int columnINDEX = column[offs++];
                    if(columnINDEX != -1){
                        VALUETYPE b   = rhs[columnINDEX];
                        tmp[row] += val * b;
                    }
                }
            }

            // store tmp                     TODO vectorise
            for(int row=0; row<C; ++row){
                lhs[chunk*C + row] = tmp[row];
            }
        }

    }

    std::vector< std::pair<int, VALUETYPE> > getRow(int const row){

        std::vector< std::pair<int, VALUETYPE> > vec;
        int const chunk (row/C);
        int const offset (row%C);
        int index = chunkOffset[chunk] + offset;


        for (int element = 0;
                element < rowLength[row]; ++element, index += C){
            vec.push_back( std::pair<int, VALUETYPE>
                            (column[index], values[index]) );
        }

        return vec;
    }

    /**
     * Row [0:N-1]; Col [0:N-1]
     */
    void addPoint(int const row, int const col, VALUETYPE value){
        if(row < 0 || col < 0 || (size_t)row >= dimension){
            throw std::invalid_argument("row and colum must be >= 0");
        }

        int const chunk (row/C);


        //// case 1: row is NOT the bigest in chunk
        if ( rowLength[row] < chunkLength[chunk] ){
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
                int lastElement = chunkOffset[chunk + 1] - C + (row%C);
                int end   = itCol - column.begin();

                for (int i = lastElement; i > end; i-=C){
                    values[i] = values[i-C];
                    column[i] = column[i-C];
                }
            }

            values[itCol - column.begin()] = value;
            *itCol = col;

            ++rowLength[row];
        }
        else{
        //// case 2: row is the longest in chunk -> expend chunk
            int const offset    = chunkOffset[chunk] + row % C;
            int const offsetEnd = chunkOffset[chunk+1];

            int index = offset;
            while (index < offsetEnd && col > column[index]){
                index += C;
            }

            if (index >= offsetEnd ){
                index = offsetEnd;

                std::vector<int>::iterator itCol = column.begin() + index;
                typename
                std::vector<VALUETYPE>::iterator itVal = values.begin() + index;

                for (int i=0; i < C; ++i){
                        itCol = column.insert(itCol, -1);   //TODO für matvecmul flag array?
                        itVal = values.insert(itVal, VALUETYPE());
                }
                *(itCol + (row%C)) = col;
                *(itVal + (row%C)) = value;
            }
            else {
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
    }

private:
    std::vector<VALUETYPE> values;
    std::vector<int>       column;
    std::vector<int>       rowLength;   // = Non Zero Entres in Row
    std::vector<int>       chunkLength; // = Max rowLength in Chunk
    std::vector<int>       chunkOffset; // COffset[i+1]=COffset[i]+CLength[i]*C
    size_t dimension;                   // = N
};

}

#endif
