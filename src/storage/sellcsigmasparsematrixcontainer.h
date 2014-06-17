#ifndef LIBGEODECOMP_STORAGE_SELL_C_SIGMA_SPARSEMATRIX_CONTAINER_H
#define LIBGEODECOMP_STORAGE_SELL_C_SIGMA_SPARSEMATRIX_CONTAINER_H

#include <vector>
#include <utility>
#include <assert.h>



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

        for (int index = chunkOffset[chunk] + offset;
                index < chunkOffset[chunk+1]; index += C){
            vec.push_back( std::pair<int, VALUETYPE>
                            (column[index], values[index]) );
        }

        return vec;
    }

    void addPoint(int const row , int const col, VALUETYPE value){
        assert(row >= 0 && col >= 0);

        int const chunk (row/C);

        if ( row > rowLength.size() ){
            rowLength.resize(row);
            chunkLength.resize(chunk);
            chunkOffset.resize(chunk);
        }


        //// Fall 1: Zeile ist nicht die längste in Chunk
        if ( rowLength[row] < chunkLength[chunk] ){
            std::vector<int>::iterator itCol = column.begin()
                    + chunkOffset[chunk] + row % C;
            //while ( col > *it && -1 != *it ){}
            while ( col > *itCol ){
                itCol += C;
            }
            assert(col != *itCol); // TODO fehler werfen? überschreiben?
            
            //// Fall 1.a Wert hinten in der Zeile einfügen auf "Auffüllwerte"
            if ( -1 == *itCol ){
                values[itCol - column.begin()] = value;
                *itCol = col;
            }
            else{
            //// Fall 1.b We)t in die mitte Einfügen
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
            std::vector<int>::iterator itCol = column.begin()
                    + chunkOffset[chunk] + row % C;
            std::vector<int>::iterator itColEnd = column.begin()
                    + chunkOffset[chunk+1] + row % C;
            while ( col > *itCol && itCol != itColEnd ){
                itCol += C;
            }
            assert(col != *itCol); // TODO fehler werfen? überschreiben?

            typename std::vector<VALUETYPE>::iterator itVal = values.begin()
                    + ( itCol - column.begin() );

            int       newCol [] = {col, -1, -1};
            VALUETYPE newVal [] = {value, VALUETYPE(), VALUETYPE()};
            //itVal = values.insert(itVal, newVal, newVal+3);
            //itCol = column.insert(itCol, newCol, newCol+3);
            itCol = column.insert(itCol, -1);

            //// fix order
            int index = itVal - values.begin();
            int i = 0;
            while ( index < chunkOffset[chunk + 1] ){
                values[index] = values[index + C];
                column[index] = column[index + C];

                if (++i < C-1){
                    ++index;
                }
                else{
                    index += 2;
                    i = 0;
                }
            }
            for (i = 0; i < C; ++i ){
                if(i != row % C){
                    values[index + i] = VALUETYPE();
                    column[index + i] = -1;
                }
            }

            ++rowLength[row];
            chunkLength[chunk] = rowLength[row];
            for (int ch = chunk; ch < chunkOffset.size(); ++ch){
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
};

}

#endif
