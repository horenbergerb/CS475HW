#ifndef MATH
#define MATH

// // // // // // // // // // // // // // // // 
//
// Some simple matrix operators.
//
//
// These routines are self contained except for random number
// generation. Comment these out if you want completely stand-alone
// functionality. Uses some code from Numerical Recipes (see
// comments). This code is NOT designed to be blindingly fast but
// rather just get the job done. It extensively checks that sizes and
// parameters are right. It intentionally draws a distiction between
// row and column vectors to help the user check their math. Any of 3
// random number generators are available to be compiled with this
// code and are shipped with this library. Pick one rand.h and one of
// the rand*.cpp. The variable Matrix::debug can be set to true if you
// want to debug memory allocation to look for overallocation of
// matrices and memory leaks. NOTE: most routines overwrite self with
// the answer. For example: add adds to self. See further in this
// comment block.
//
// Author: Dr. Robert B. Heckendorn, University of Idaho
// Version: 3.4W
// Date: Feb 27, 2019
//
// You are free to use, change, or redistribute the code in any way
// you wish for non-commercial purposes, but please maintain the name
// of the original author. This code is not guarantteed to function
// correctly and comes with no warranty of any kind for any purpose
// what-so-ever and is NOT SUPPORTED.   Good luck and have fun.

// IMPORTANT: If running on MICROSOFT WINDOWS uncomment this define statement!
// #define WINDOWS

// 
// WARNING: Most matrix library routines REPLACE the contents of the matrix
// object (overwrite self).  That is: X.sub(Y) will replace X with X - Y.
// Rule 1: If the routine returns "Matrix &" then it probably modifies the matrix.
// Rule 2: If the routine returns "Matrix" then it creates a new matrix and so the
// value needs to saved, printed, or stored.
//
// Some routines allocate a new matrix leaving the original
// matrix untouched.  For these routines you often need to assign the result
// to a new matrix: X = Y.dot(Z)
// or print them out: Y.dot(Z).print()
// 

#include <stdlib.h>
#include <stdio.h>
#include <vector>       // supports submatrices
#include <string>       // matrix names are strings
#include "rand.h"       // portable random number generator.  Include exactly
                        // ONE of the random number cpp files in your compile
// #define WALSH           // activate the Walsh library by defining this symbol

static const double EPSILONOFZERO=1E-8;

class Matrix;

// helper function
int bitCount(unsigned int w);

// // // // // // // // // // // // // // // // 
//
// class MatrixRowIter
//
// Iterator for incrementing through rows in a matrix
//
class MatrixRowIter {
private:
    Matrix *mat;
    int r;
    Matrix *arow;
    bool more;

public:
    MatrixRowIter(Matrix *mat);
    ~MatrixRowIter();

public:
    Matrix *rowBegin();
    Matrix *rowNext();
    bool rowNotEnd();
    int row();
};



// // // // // // // // // // // // // // // // 
//
// class Matrix
//
// A simple class for matrix operations.  It has some nice debugging
// features like trying hard to check that the proper dimensions are used.
// It is very draconian about this so there is an important different
// between row vectors and column vectors.  I find this helps students get
// the math to work out correctly if you pay attention to this difference.
// The routines allow you to name a matrix.  The name is then used in debug
// output.  Other things checked include referencing out of bounds.
//

class Matrix {
friend class MatrixRowIter;
public:
    static bool debug;      // debugging flag

private:
    bool defined;           // does it have rows and cols defined
    bool submatrix;         // if submatrix then it does NOT own the row content of m (see deallocate)!!
    int maxr, maxc;         // number of rows and columns
    double **m;             // the data
    std::string name;       // the name of the matrix or ""

private:  // private methods
    void allocate(int r, int c, std::string namex, bool isSubMatrix=false);
    bool deallocate();
    void reallocate(int othermaxr, int othermaxc, std::string namex);

private:
    static char *realFormat;
    static char *intFormat;
    static char *shortIntFormat;

// constructors
public:
    Matrix(std::string namex="");
    Matrix(int r, std::string namex="");                            // create a subMatrix columns unallocated
    Matrix(int r, int c, std::string namex="");
    Matrix(int r, int c, double initValue, std::string namex="");   // create and initialize
    Matrix(int r, int c, double *data, std::string namex="");       // create and initialize from array
    Matrix(const Matrix &other, std::string namex="");              // copy constructor
    Matrix(Matrix *other);                                          // for convenience
    ~Matrix();
    Matrix &operator=(const Matrix &other);

// basic error checking support
public:
    void checkBounds(int r, int c, std::string msg) const;
    void assertColVector(std::string) const;
    void assertRowsEqual(const Matrix &other, std::string msg) const;
    void assertColsEqual(const Matrix &other, std::string msg) const;
    void assertDefined(std::string msg) const;
    void assertRowIndexOK(int r, std::string msg) const;
    void assertColIndexOK(int c, std::string msg) const;
    void assertIndexOK(int, int, std::string) const;
    void assertOtherLhs(const Matrix &other, std::string msg) const;
    void assertOtherSizeMatch(const Matrix &other, std::string msg) const;
    void assertRowVector(std::string) const;
    void assertSize(int r, int c, std::string msg) const;
    void assertUsableSize(std::string msg) const;
    void assertSquare(std::string msg) const;
    void assertRowPower2(std::string msg) const;
    void assertColPower2(std::string msg) const;

public:  // auxillary routines but not private (for speed, they do not check self!!)
    void swapRows(int i, int j);                  // utility to swap two rows
    bool lessRows(int i, int j) const;            // utility to compare two rows

// accessors
public: 
    int numRows() const { return maxr; }
    int numCols() const { return maxc; }
    double get(int r, int c) const;      // get element value
    double inc(int r, int c);            // increment element
    double dec(int r, int c);            // decrement element
    double set(int r, int c, double v);  // set element
    void setDefined();                   // make defined when you *KNOW* the array has been defined by other means
    void setName(std::string newName);   // set matrix name
    const std::string &getName(const std::string &defaultName="") const;
    void narrow(int newMaxCol);          // remove trailing columns (without proper deallocation)
    void widen(int newc, double fill=0.0); // widen the matrix filling with constant
    void shorten(int newMaxRow);         // remove trailing rows (without proper deallocation)
    void lengthen(int newc, double fill=0.0); // lengthen the matrix filling with constant

// Boolean tests
public: 
    bool isDefined() const { return defined; }               // return true if defined
    bool isRowVector() const { return defined && maxr==1; }  // exactly one row
    bool isColVector() const { return defined && maxc==1; }  // exactly one col
    bool equal(const Matrix &other) const;       // are the two matrices equal?
    bool nearEqual(double epsilon, const Matrix &other) const; // matrices nearly equal?

// basic properties
public:    
    int countGreater(const Matrix &other) const; // count number of elements >
    void argMax(int &r, int &c) const;           // what location is the largest in whole array
    void argMin(int &r, int &c) const;           // what location is the smallest in whole array
    Matrix argMaxRow() const;                    // constructs a column vector of the argmax in each row
    Matrix argMinRow() const;                    // constructs a column vector of the argmin in each row
    Matrix minRow() const;                       // constructs a column vector of the min in each row
    double max() const;                          // minimum in whole array
    double min() const;                          // maximum in whole array
    double mean() const;                         // mean of whole array
    double stddev() const;                       // standard deviation of whole array
    double maxCol(int c) const;                  // maximum value in a column
    double minCol(int c) const;                  // minimum value in a column
    double meanCol(int c) const;                 // mean in a column
    double stddevCol(int c) const;               // standard deviation in a column
    int countEqCol(int c, double value) const;   // count number of items in column c equal to value
    int countNeqCol(int c, double value) const;  // count number of items in column c not equal to value
    double dot(int r, int c, const Matrix &other) const;  // dot of row of this with col of other -> double 

    // lengths and distances (beware that some distances are squares of distances
    double sum() const;                          // sums up elements in the matrix
    double dist2() const;                        // sums up squares of elements matrix
    Matrix dist2Row() const;                     // square of length of each rows -> col vector
    double dist2(const Matrix &other) const;     // *SQUARE* of distance between two matrices
    double dist2(int r, int c, const Matrix &other) const;  // *SQUARE* of distance between row of this with col of other

    // element by element operators (modifies self)
    Matrix &abs();
    Matrix &add(const Matrix &other);
    Matrix &sub(const Matrix &other);
    Matrix &mul(const Matrix &other);
    Matrix &div(const Matrix &other);

    Matrix &swap(Matrix &other);    // swaps two matrices so also modifies other
    Matrix &rowInc(int r);          // increment the values in a given row by 1

    // initialize to constants (obviously modifies self)
    Matrix &constant(double x);            // this can be used to zero a matrix
    Matrix &constantDiagonal(double x);    // this can be used to set the diagonal to a constant but does set rest of matrix
    Matrix &constantCol(int c, double x);  // this can be used to zero a column
    Matrix &constantColRange(int c, double start, double step);   // assign all elements in col range starting at start and going by step
    Matrix &identity();                    // convert to an identity matrix  (must be square)

    // random initialization (random number generator must be initialized with initRand() )
    // (obviously modifies self)
    Matrix &randCol(int c, double min, double max);  // random reals in given column
    Matrix &randNorm(double mean, double stddev);    // random reals in normal distribution
    Matrix &rand(double min, double max);            // random reals in range 
    Matrix &rand(int min, int max);                  // random ints in range (doesn't include max)

    // initialization by formula (obviously modifies self)
    Matrix &initLinear(double A, double B, double C); // init m[r][c] = A*r + B*c + C;

    // scalar operators (modifies self)
    Matrix &scalarMul(double x);          // multiply all elements by x
    Matrix &scalarAdd(double x);           // add to all elements x
    Matrix &scalarPreSub(double x);        // NOTE: this is x - self   not   self - x, can be used to negate
    Matrix &scalarPostSub(double x);       // NOTE: this is self - x

    // Vector operations (modifies self)
    Matrix &divColVector(const Matrix &other);
    Matrix &mulColVector(const Matrix &other);

    Matrix &divRowVector(const Matrix &other);  // self[r] / (row vector other) for each row
    Matrix &mulRowVector(const Matrix &other); // self[r] * (row vector other) for each row
    Matrix &addRowVector(const Matrix &other);  // self[r] + (row vector other) for each row
    Matrix &subRowVector(const Matrix &other);  // self[r] - (row vector other) for each row
    Matrix &addRowVector(int r, const Matrix &other); // add row vector matrix in other to the given row of self
    
    // min/max normalization by columns (MODIFIES SELF)
    void normalize();                             // normalize matrix so values fall between 0 and 1
    Matrix normalizeCols();                       // normalize columns in place and return array of min and max of each col
    Matrix &normalizeCols(Matrix &minMax);        // normalize based on an array of min and max for each col
    void unnormalizeCols(Matrix &minMax);         // undo the normalization done by normalizeCols using the minMax matrix

    // mapping functions  (modifies self)
    Matrix &map(double (*f)(double x));              // apply given function to all elements
    Matrix &mapCol(int c, double (*f)(double x));    // apply given function to all elements in col c
    Matrix &mapIndex(double (*f)(int r, int c, double x)); // apply function to (index, element)
    // creates new matrix
    Matrix mapCol(double (*f)(int size, double *x)); // apply function to each column -> one double
    Matrix mapRow(double (*f)(int size, double *x)); // apply function to each row -> one double
    Matrix cartesianRow(double (*)(int, double*, double*), Matrix&);  // apply given function to the cartesian product of two vectors of row vectors
    Matrix seriesSampleCol(int col, int numsteps, int stride);   // sample a column as if a time series

    // random actions
    Matrix &sample(Matrix &out);  // extract random rows with replacement into existing matrix out
    Matrix &shuffle();            // randomly shuffle the rows (changes self)

    // insertion and extraction
    Matrix &extract(int minr, int minc, int sizer, int sizec, Matrix &out);  // extract into existing matrix out (see other versions of extract)
    Matrix &insert(const Matrix &other, int minr, int minc);    // insert the matrix at minr, minc.   Overflow is ignored.
    Matrix &insertRowVector(int row, const Matrix&);
    Matrix pickRows(int match, const Matrix &list);   // pick rows i in which list[i]==match
    Matrix joinRight(Matrix &other);                 // joins other to the right of self giving new matrix
    
    // input/output
    Matrix &printfmt(std::string msg="", std::string fmt=realFormat, bool size=false);  // print matrix and optionally a msg with numbers in given format and with or without size
    Matrix &print(std::string msg="");         // print matrix and its name (returns arg for pipes)
    void printInt(std::string msg="") const;   // print matrix as integers (it will error if not.)
    void printNZ(double epsilon=EPSILONOFZERO, std::string msg="") const;    // print matrix zeroing anything that is near zero
    void printLabeledRow(char **labels, std::string msg="") const;   // print matrix with rows labeled
    void printSize(std::string msg="") const;  // print just the matrix size
    void write();                        // write out matrix in a form that can be read in
    void writeLine(int r);               // write out an unadorned row
    void read();                         // read in a matrix
    void readT();                        // read in a matrix and transpose it (just read that way)
    char **readLabeledRow();             // read in a matrix plus row labels

private:
    char **readAux(bool labeled, bool transpose);    // general read routine both labeled and un

#ifdef WALSH
#include "matwalsh.h"
#endif
    
public:
    // WARNING: The following *CONSTRUCT TO NEW MATRIX* for the answer  (BEWARE MEMORY LEAKS!)
    // WARNING: the result of these functions should be used somewhere like in an assignment
    //  e.g. a.dot(b) is PROBABLY WRONG.   while x = a.dot(b) is CORRECT and stores the result.
    Matrix extract(int minr, int minc, int sizer, int sizec);
    Matrix extractStride(int minr, int minc, int stepr, int stepc);
    Matrix transpose();                    // classic transpose into NEW MATRIX (see transposeSelf below)

    Matrix dot(const Matrix &other);       // classic matrix multiply, inner product
    Matrix dotT(const Matrix &other);      // classic matrix multiply self * Transpose(other)
    Matrix Tdot(const Matrix &other);      // classic matrix multiply Transpose(self) * other
    Matrix meanVec();                      // creates a row vector of means of columns
    Matrix stddevVec();                    // creates a row vector of standard deviations of columns
    Matrix cov();                          // covariance matrix (BIASED covariance)
    Matrix cov(Matrix &other);             // covariance matrix (BIASED covariance)

    Matrix &transposeSelf();                // transpose in place of SQUARE MATRIX

    // special operators (destroys arguments)
    int *LU();                              // LU decomposition in place
    Matrix &solve(Matrix &B);               // solve Ax = B returns solutions and inverse
    Matrix &inverse();                      // replace with inverse

    // eigenSystem() destroys self by replacing self with eigenvectors in rows.
    // Returns a new matrix with the eigenvalues in it.
    // Eigenvalues and vectors returned sorted from largest magnitude to smallest
    // WARNING: allocates new matrix for answer
    void tridiagonalize(double *&d, double *&e);
    Matrix eigenSystem();

    // sorting support
private:
    void selectSort(int lower, int upper);
    void qs(int lower, int upper);
    void selectSortCol(int c, int lower, int upper);
    void qsCol(int c, int lower, int upper);

    // sorting
public: 
    void sortRows();                            // sort rows in place
    void sortRows(int startRow, int endRow);    // sort rows in place in a range of rows
    void sortRowsByCol(int c);                  // sort rows in place on given column
    void sortRowsByCol(int c, int startRow, int endRow);     // sort rows in place in a range of rows

public:
    // subMatrices are here for EFFICIENCY for creating submatrices by pointing into
    // the parent matrix rather than copying all the contents of the matrix.
    // READ THE WARNINGS in the .cpp file
    // 
    Matrix subMatrix(int minr, int minc, int sizer, int sizec) const;  // create a submatrix whose corner is (minr, minc) and size given
    Matrix subMatrixEq(int c, double value) const;         // create submatrix with rows whose column c has the given value
    Matrix subMatrixNeq(int c, double value) const;        // create submatrix with rows whose column c does not have the given value

    // image (picture) support (currently only supports 8 bit pgm and ppm formats)
    // output is in ascii formats (zzz: fix someday to use more compressed output)
    // 8 bit gray is one integer in the range 0-255 for each pixel
    // 8 bit color is three integers in a row in the range 0-255 for RGB in each pixel.
    // That is an 8 bit color square 100x100 pixels gens a 100x300 dimensional array
private:
    int byteValue(double x);
    Matrix readImage(char *expectedType, char *caller, std::string filename, std::string namex);

public: 
    Matrix readImagePgm(std::string filename, std::string namex);   // read a P2 or P5 pgm  (8 bit gray scale) file into self
    Matrix readImagePpm(std::string filename, std::string namex);   // read a P3 or P6 ppm  (8 bit color)
    void writeImagePgm(std::string filename, std::string comment);  // write a P2 pgm file  (8 bit gray scale)
    void writeImagePpm(std::string filename, std::string comment);  // write a P3 pgm file (8 bit color)
};

#endif

