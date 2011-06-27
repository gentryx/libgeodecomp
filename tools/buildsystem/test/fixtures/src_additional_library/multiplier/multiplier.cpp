#include "multiplier.h"

std::string
Multiplier::mult(std::string message, int n) {
    std::string ret = "";
    for (int i = 0; i < n; i++)
        ret += message;
    return ret;
}
