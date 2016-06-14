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
    char b = f ^ x;
    return b ^ (b >> 1);
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

void _read_row_slice(FILE* fp, int* shape, int row, Slice* col, long* matrix)
{
    int c;
    ItemIdx idx;
    for (c = col->start; c < col->stop; c += col->step)
    {
        idx.r = row;
        idx.c = c;
        matrix[c - col->start] = (long) _read_item(fp, shape, &idx);
    }
}

void _read_col_slice(FILE* fp, int* shape, int col, Slice* row, long* matrix)
{
    int r;
    ItemIdx idx;
    for (r = row->start; r < row->stop; r += row->step)
    {
        idx.r = r;
        idx.c = col;
        matrix[r - row->start] = (long) _read_item(fp, shape, &idx);
    }
}

void
bed_read_row_slice(char* filepath, int nrows, int ncols, int row,
                   int c_start, int c_stop, int c_step,
                   long* matrix)
{
    int shape[2] = {nrows, ncols};
    Slice cslice = {c_start, c_stop, c_step};

    FILE* fp = fopen(filepath, "rb");
    _read_row_slice(fp, shape, row, &cslice, matrix);
    fclose(fp);
}

void
bed_read_col_slice(char* filepath, int nrows, int ncols, int col,
                   int r_start, int r_stop, int r_step,
                   long* matrix)
{
    int shape[2] = {nrows, ncols};
    Slice rslice = {r_start, r_stop, r_step};

    FILE* fp = fopen(filepath, "rb");
    _read_col_slice(fp, shape, col, &rslice, matrix);
    fclose(fp);
}

void
bed_read_slice(char* filepath, int nrows, int ncols,
           int r_start, int r_stop, int r_step,
           int c_start, int c_stop, int c_step,
           long* matrix)
{
    int shape[2] = {nrows, ncols};
    Slice rslice = {r_start, r_stop, r_step};
    Slice cslice = {c_start, c_stop, c_step};

    FILE* fp = fopen(filepath, "rb");
    _read_slice(fp, shape, &rslice, &cslice, matrix);
    fclose(fp);
}

void
bed_read(char* filepath, int nrows, int ncols, long* matrix)
{
    bed_read_slice(filepath, nrows, ncols, 0, nrows, 1, 0, ncols, 1, matrix);
}

int
bed_read_item(char* filepath, int nrows, int ncols, int row, int col)
{
    int shape[2] = {nrows, ncols};
    ItemIdx idx = {row, col};

    FILE* fp = fopen(filepath, "rb");
    int item = _read_item(fp, shape, &idx);
    fclose(fp);

    return item;
}

int
bed_major(char* filepath)
{
    FILE* fp = fopen(filepath, "rb");
    fseek(fp, 2, SEEK_SET);
    char item = fgetc(fp);
    fclose(fp);

    return item;
    // 00000000 individual major: SNP 1, SNP 2, ...
    // 00000001 SNP major: individual 1, individual 2, ...
}
