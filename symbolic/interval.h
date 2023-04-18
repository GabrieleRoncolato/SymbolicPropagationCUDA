//
// Created by server on 1/24/23.
//

#ifndef TEST_INTERVAL_H
#define TEST_INTERVAL_H

#include "matrix.h"

/* define the structure of interval */
struct Interval
{
    struct Matrix lower_matrix;
    struct Matrix upper_matrix;
};

#endif //TEST_INTERVAL_H
