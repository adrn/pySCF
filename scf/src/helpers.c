int getIndex2D(int row, int col, int ncol) {
    return row*ncol + col;
}

int getIndex3D(int row, int col, int dep, int ncol, int ndep) {
    return (row*ncol + col)*ndep + dep;
}
