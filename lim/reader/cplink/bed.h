#ifndef BED_H
#define BED_H

#include "stdio.h"

int
read_item(char* filepath, int nrows, int ncols, int row, int col);

void
read_slice(char* filepath, int nrows, int ncols,
           int r_start, int r_stop, int r_step,
           int c_start, int c_stop, int c_step,
           long* matrix);

#endif
