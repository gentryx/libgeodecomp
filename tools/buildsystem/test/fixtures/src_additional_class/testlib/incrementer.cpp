#include <iostream>
#include "incrementer.h"

void
Incrementer::inc(int i) {
    std::cout << "Incrementer::inc(" << i << ") = " << (i+1) << "\n";
}
