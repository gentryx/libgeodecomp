#ifndef LIBGEODECOMP_STORAGE_SELL_C_SIGMA_SPARSEMATRIX_CONTAINER_H
#define LIBGEODECOMP_STORAGE_SELL_C_SIGMA_SPARSEMATRIX_CONTAINER_H

#include <vector>
#include <utility>
#include <assert.h>

#include <iostream>




namespace LibGeoDecomp {


//OHNE SORTIEREN! SIGMA =1
//NICHT PARRALEL!
template<typename VALUETYPE, int C = 1, int SIGMA = 1>
class SellCSigmaSparseMatrixContainer
{
public:
    SellCSigmaSparseMatrixContainer(){}

    void matVecMul (std::vector<double> lhs, std::vector<double> rhs){
        //TODO
    }

    std::vector< std::pair<int, VALUETYPE> > getRow(int const row){

        std::vector< std::pair<int, VALUETYPE> > vec;
        int const chunk (row/C);
        int const offset (row%C);

std::cout << "Get row: row="<< row << " chunk=" << chunk << " offset=" << offset << " chunkOffset=" << chunkOffset[chunk] << " Werte=";
        for (int index = chunkOffset[chunk] + offset;
                index < chunkOffset[chunk+1]; index += C){
            vec.push_back( std::pair<int, VALUETYPE>
                            (column[index], values[index]) );
std::cout << "(" << vec.back().first << ","<< vec.back().second << "), ";
        }
std::cout << std::endl;
        return vec;
    }

    // Row [0:N-1]; Col [0:N-1]
    void addPoint(int const row , int const col, VALUETYPE value){
        assert(row >= 0 && col >= 0);

        int const chunk (row/C);

std::cout << "Add point: row: " << row << " chunk: " << chunk << " col=" << col << " val=" << value;

        if ( row >= rowLength.size() ){
//std::cout << "rezie row" << std::endl;
            rowLength.resize(row+1);
        }
        if ( chunk >= chunkLength.size() ){
//std::cout << "rezie chunk" << std::endl;
            int oldNumberOfChunks = chunkLength.size();

            chunkLength.resize(chunk+1);
            chunkOffset.resize(chunk+2);

            for( int i = oldNumberOfChunks; i < chunkOffset.size(); ++i ){
                chunkOffset[i] = values.size();
            }
        }

std::cout << " row length: " << rowLength[row] << " chunk Length: " << chunkLength[chunk] << " chunkOffset: " << chunkOffset[chunk] << std::endl;

        //// Fall 1: Zeile ist nicht die längste in Chunk
        if ( rowLength[row] < chunkLength[chunk] ){
std::cout << "fall 1";
            std::vector<int>::iterator itCol = column.begin()
                    + chunkOffset[chunk] + row % C;

            while ( col > *itCol && -1 != *itCol ){
                itCol += C;
            }
            assert(col != *itCol); // TODO fehler werfen? überschreiben?
            
            //// Fall 1.a Wert hinten in der Zeile einfügen auf "Auffüllwerte"
            if ( -1 == *itCol ){
std::cout << "a" << std::endl;
                values[itCol - column.begin()] = value;
                *itCol = col;
            }
            else{
            //// Fall 1.b Wert in die mitte Einfügen TODO fixen
std::cout << "b" << std::endl;
                int index = itCol - column.begin();

                for (int i = index + C; i < chunkOffset[chunk +1]; i+=C){
                    values[i] = values[i-C];
                    column[i] = column[i-C];

                    //if ( column[i+C] == -1 )
                        //break;
                }

                values[index] = value;
                column[index] = col;
            }

            ++rowLength[row];
        }
        else{
        //// Fall 2: Zeile ist die Längste in Chunk
std::cout << "fall 2";

            int offset = chunkOffset[chunk] + row % C;
            int offsetEnd = chunkOffset[chunk+1];

            while (offset < offsetEnd && col > column[offset]){
                offset += C;
            }
//std::cout << "offset:" << offset << " offsetEnd:" << offsetEnd << std::endl;
            if (offset >= offsetEnd ){ //TODO schöner mit einem fall schreiben
std::cout << "a" << std::endl;
                offset = offsetEnd;

                std::vector<int>::iterator itCol = column.begin() + offset;
                typename
                std::vector<VALUETYPE>::iterator itVal = values.begin() + offset;

                for (int i=C; i > 0; --i){
                        itCol = column.insert(itCol, -1);               
                        itVal = values.insert(itVal, VALUETYPE());
                }
                *(itCol + (row%C)) = col;
                *(itVal + (row%C)) = value;
            }
            else if (offset < offsetEnd ){
std::cout << "b" << std::endl;
                assert(col != column[offset]); //TODO fehler werfen?

                std::vector<int>::iterator itCol = column.begin() + offset;
                typename
                std::vector<VALUETYPE>::iterator itVal = values.begin() + offset;

                for (int i=0; i < C; ++i){
                    itCol = column.insert(itCol, -1);               
                    itVal = values.insert(itVal, VALUETYPE());
                }
                *itVal = value;
                *itCol = col;

                //// fix order
                int index = itVal - values.begin();
                int i = 0;
                while ( index < chunkOffset[chunk + 1] ){
                    ++index;

                    if (i < C-1){
                        ++i;
                    }
                    else{
                        i = 0;
                        continue;
                    }

                    values[index] = values[index + C];
                    column[index] = column[index + C];
                }
                for (i = 0; i < C; ++i ){
                    if(i != row % C){
                        values[index + i] = VALUETYPE();
                        column[index + i] = -1;
                    }
                }
            }

            ++rowLength[row];
            chunkLength[chunk] = rowLength[row];
            for (int ch = chunk+1; ch < chunkOffset.size(); ++ch){
                chunkOffset[ch] += C;
            }
        }

std::cout << "col: ";
for (int i=0; i<column.size(); ++i){
    std::cout << column[i] << " ";
}
std::cout << std::endl;
std::cout << "values: ";
for (int i=0; i<values.size(); ++i){
    std::cout << values[i] << " ";
}
std::cout << std::endl;

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
