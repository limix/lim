#include "stdio.h"

int FILE_OFFSET = 3;

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

int row_size(int shape[2])
{
    return shape[1] / 4;
}

void convert_idx_itby(int shape[2], ItemIdx* idx, ByteIdx* bydx)
{
    bydx->s = FILE_OFFSET + (shape[1] / 4) * idx->r + idx->c / 4;
}

void convert_idx_itbi(int shape[2], ItemIdx* idx, BitIdx* bidx)
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

void read_item(FILE* fp, int shape[2], ItemIdx* idx)
{
    ByteIdx bydx;
    convert_idx_itby(shape, idx, &bydx);

    BitIdx bidx;
    convert_idx_itbi(shape, idx, &bidx);

    printf("byte idx %d\n", bydx.s);
    printf("bit idx %d\n", bidx.s);

    fseek(fp, bydx.s, SEEK_SET);
    char v = fgetc(fp);
    printf("SNP %d\n", get_snp(v, &bidx));
}

int main()
{
    char* path = "/Users/horta/workspace/lim/lim/data/plink/example/test.bed";
    FILE* bed = fopen(path, "rb");
    int shape[2] = {5, 6};
    printf("shape %d %d\n", shape[0], shape[1]);

    ItemIdx idx = {0, 0};
    read_item(bed, shape, &idx);
    idx.c = 1;
    read_item(bed, shape, &idx);
    idx.c = 2;
    read_item(bed, shape, &idx);
    idx.c = 3;
    read_item(bed, shape, &idx);

    fclose(bed);
}
