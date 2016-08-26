from os.path import join
from os.path import dirname
from os.path import relpath
from os.path import realpath

from cffi import FFI

root = relpath(dirname(realpath(__file__)))

ffi = FFI()

include_dirs = [root]
src_files = [join(root, 'bed.c')]

ffi.set_source('lim.reader.cplink.bed_ffi',
               '#include "bed.h"',
               include_dirs=include_dirs,
               sources=src_files,
               libraries=[])

ffi.cdef("""
int bed_read_item(char* filepath, int nrows, int ncols, int row, int col);

void
bed_read_slice(char* filepath, int nrows, int ncols,
               int r_start, int r_stop, int r_step,
               int c_start, int c_stop, int c_step,
               long* matrix);

void
bed_read_row_slice(char* filepath, int nrows, int ncols, int row,
                   int c_start, int c_stop, int c_step,
                   long* matrix);

void
bed_read_col_slice(char* filepath, int nrows, int ncols, int col,
                  int r_start, int r_stop, int r_step,
                  long* matrix);

void
bed_read(char* filepath, int nrows, int ncols, long* matrix);

int bed_major(char* filepath);
""")

if __name__ == '__main__':
    ffi.compile(verbose=True)
