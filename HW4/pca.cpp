#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "rand.h"
#include "mat.h"

/*
What's the planiel, staniel?

take in R samples of C-dimensional vectors (an image in RxC)
take in an integer (eigenvectors to keep)
    If it's negative, transpose the matrix, and untranspose when you're done
Use PCA
    Center columns about the mean
    Calculate covariance matrix
    Compute eigenvalues, eigenvectors
    Normalize eigenvectors
    Sort eigenvectors by eigenvalue
    Take the K largest (remove the rest)
Get the image back
    rotate back: compressed image.dot(selectedEigenVectors)
    add the mean to all the columns
    "OR IF Z-SCORE" what does that mean?
    
Compare the sizes of encoded and decoded
Do sum of squares between original and new pixels
save the new file as z.ppm or z.pgm (depending on whether the image is in color)
 */

int main(int argc, char* argv[]){

  int negVal = 0;
  
  int k = 100;
  if(argc == 2)
    k = atoi(argv[1]);
  else{
    printf("You messed up\n");
    exit(0);
  }

  FILE *proxyFile;
  void *content = malloc(sizeof(char)*100);
  int read;
  proxyFile = fopen("proxyFile.txt", "w");
  while((read = fread(content, 1, sizeof(char)*100, stdin)))
    fwrite(content, read, 1, proxyFile);
  fclose(proxyFile);
  
  
  //Read in the image
  bool isColor;
  Matrix rawImage;
  
  rawImage.readImagePixmap("proxyFile.txt","Image", isColor);

  if(k < 0){
    negVal = 1;
    k = k*-1;
    rawImage = rawImage.transpose();
  }
  rawImage.printSize();
  
  //First we find the mean and subtract from the rows
  Matrix meanValues = rawImage.meanVec();
  meanValues.setName("Mean");

  meanValues.printSize();
  
  rawImage.addRowVector(meanValues.scalarMul(-1.0));

  //Computing the covariance used to find eigens
  Matrix eigenVectors = rawImage.cov();

  //rawImage = eigenVectors.subMatrix(0, 0, eigenVectors.numRows(), eigenVectors.numCols());
  
  //Turns the covariance into eigenvectors and values (ALREADY SORTED!)
  Matrix eigenValues = eigenVectors.eigenSystem();
  eigenValues.setName("EigenValues");
  eigenVectors.setName("EigenVectors");
  eigenVectors.printSize();
  eigenValues.printSize();

  Matrix kVectors = eigenVectors.subMatrix(0,0, k, eigenVectors.numCols());
  Matrix kValues = eigenValues.subMatrix(0,0, eigenValues.numRows(), k);

  Matrix compressedImage = rawImage.dot(kVectors.transpose());
  compressedImage.setName("Encoded");
  compressedImage.printSize();
  compressedImage = compressedImage.dot(kVectors);
  compressedImage.addRowVector(meanValues.scalarMul(-1.0));
  compressedImage.setName("Decoded");
  compressedImage.printSize();

  rawImage.addRowVector(meanValues);
  
  double distance = compressedImage.dist2(rawImage);
  distance = distance/(double)(rawImage.numRows()*rawImage.numCols());

  printf("Per Pixel Dist^2: %f", distance);

  if(negVal){
    compressedImage = compressedImage.transpose();
  }
  
  if(!isColor)
    compressedImage.writeImagePgm("z.pgm", "");
  else
    compressedImage.writeImagePpm("z.ppm", "");

  remove("proxyFile.txt");
}


