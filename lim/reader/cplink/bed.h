#ifndef BED_H
#define BED_H

#include "stdio.h"

int
bed_read_item(char* filepath, int nrows, int ncols, int row, int col);

void
bed_read_slice(char* filepath, int nrows, int ncols,
           int r_start, int r_stop, int r_step,
           int c_start, int c_stop, int c_step,
           long* matrix);

void
bed_read(char* filepath, int nrows, int ncols, long* matrix);

void
bed_read_row_slice(char* filepath, int nrows, int ncols, int row,
                   int c_start, int c_stop, int c_step,
                   long* matrix);

void
bed_read_col_slice(char* filepath, int nrows, int ncols, int col,
                  int r_start, int r_stop, int r_step,
                  long* matrix);

int bed_major(char* filepath);

#endif
