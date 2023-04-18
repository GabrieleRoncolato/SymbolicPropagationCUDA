//
// Created by server on 1/24/23.
//
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#ifndef TEST_MATRIX_H
#define TEST_MATRIX_H


/* Define the structure of Matrix */
struct Matrix
{
    float* data;
    int row, col;
};


/* add the constant to matrix */
void add_constant(struct Matrix* A, float alpha);


/* element-wise multiplication */
void multiply(struct Matrix* A, struct Matrix* B);


/* print matrix */
void printMatrix(struct Matrix* A);


/* print matrix to the file */
void fprintMatrix(FILE *fp, struct Matrix* A);


/* takes the relu of the matrix */
void relu(struct Matrix* A);

#endif