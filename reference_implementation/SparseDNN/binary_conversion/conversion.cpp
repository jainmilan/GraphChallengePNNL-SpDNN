#include <stdio.h>

int main(void) {
    // example of 1024 (4095, 16384, 65536 can replace)
    FILE *fp = fopen("sparse-images-1024.tsv", "r");
    int row_count = 6374505; // wc -l sparse-images-1024.tsv
    int *row = new int[row_count];
    int *col = new int[row_count];
    int *val = new int[row_count];
    for (int i = 0; i < row_count; i++) {
        fscanf(fp, "%d\t%d\t%d\n", col, row, val);
        col += 1;
        row += 1;
        val += 1;
    }
    fclose(fp);

    fp = fopen("sparse-images-1024.bin", "wb");
    fwrite(row, sizeof(int), row_count, fp);
    fwrite(col, sizeof(int), row_count, fp);
    fwrite(val, sizeof(int), row_count, fp);
    fclose(fp);
}
