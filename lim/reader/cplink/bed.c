#include "bed.h"

int FILE_OFFSET = 3;

typedef struct
{
    int start;
    int stop;
    int step;
} Slice;

typedef struct
{
    int r;
    int c;
} ItemIdx;

typedef struct
{
    int s;
} ByteIdx;

typedef struct
{
    int s;
} BitIdx;

int row_size(int* shape)
{
    return shape[1] / 4;
}

void convert_idx_itby(int* shape, ItemIdx* idx, ByteIdx* bydx)
{
    bydx->s = FILE_OFFSET + ((shape[1] + 3)/ 4) * idx->r + idx->c / 4;
}

void convert_idx_itbi(int* shape, ItemIdx* idx, BitIdx* bidx)
{
    bidx->s = (idx->c % 4) * 2;
}

char get_snp(char v, BitIdx* bidx)
{
    char f = (v >> bidx->s) & 3;
    char bit1 =  f & 1;
    char bit2 =  (f >> 1) & 1;
    char x = (bit1 ^ bit2);
    x = (x << 0) | (x << 1);
    return f ^ x;
}

char _read_item(FILE* fp, int* shape, ItemIdx* idx)
{
    ByteIdx bydx;
    convert_idx_itby(shape, idx, &bydx);

    BitIdx bidx;
    convert_idx_itbi(shape, idx, &bidx);

    fseek(fp, bydx.s, SEEK_SET);

    char item = get_snp(fgetc(fp), &bidx);

    return item;
}

void _read_slice(FILE* fp, int* shape, Slice* row, Slice* col, long* matrix)
{
    int ri = 0, ci;
    int r, c;
    ItemIdx idx;
    int ncols_read = (col->stop - col->start) / col->step;
    for (r = row->start; r < row->stop; r += row->step)
    {
        ci = 0;
        for (c = col->start; c < col->stop; c += col->step)
        {
            idx.r = r;
            idx.c = c;
            matrix[ri * ncols_read + ci] = (long) _read_item(fp, shape, &idx);
            ci++;
        }
        ri++;
    }
}

void
read_slice(char* filepath, int nrows, int ncols,
           int r_start, int r_stop, int r_step,
           int c_start, int c_stop, int c_step,
           long* matrix)
{
    int shape[2] = {nrows, ncols};
    Slice rslice = {r_start, r_stop, r_step};
    Slice cslice = {c_start, c_stop, c_step};

    int nrows_read = (r_stop - r_start) / r_step;
    int ncols_read = (c_stop - c_start) / c_step;

    FILE* fp = fopen(filepath, "rb");
    _read_slice(fp, shape, &rslice, &cslice, matrix);
    fclose(fp);
}

int
read_item(char* filepath, int nrows, int ncols, int row, int col)
{
    int shape[2] = {nrows, ncols};
    ItemIdx idx = {row, col};

    FILE* fp = fopen(filepath, "rb");
    int item = _read_item(fp, shape, &idx);
    fclose(fp);

    return item;
}
