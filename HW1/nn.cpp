#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <fstream>
#include "rand.h"
#include "mat.h"


//NOTE: change trainDataOutputs to testDataInputs

double activationFunc(double x);
static double actSlope = .7;

int main()
{
  int inputs;
  scanf("%d", &inputs);

  Matrix trainData;
  trainData.read();
  Matrix testData;
  testData.read();

  //Creating a matrix which separates the inputs and outputs
  Matrix trainDataInputs = trainData.subMatrix(0, 0, trainData.numRows(), inputs);

  Matrix trainDataOutputs = trainData.subMatrix(0, inputs, trainData.numRows(), trainData.numCols()-inputs);

  //Adding a bias
  Matrix inputsAndBias = new Matrix(trainDataInputs.numRows(),trainDataInputs.numCols()+1, -1.0);
  Matrix outputsAndBias = new Matrix(testData.numRows(), testData.numCols()+1, -1.0);
  inputsAndBias.insert(trainDataInputs, 0, 0);
  //inputsAndBias.constantCol(inputsAndBias.numCols()-1, -1.0);
  outputsAndBias.insert(testData, 0, 0);

  inputs = inputs+1;
  //Next we're going to train on the test data (matrix 1)

  int iterations = 10000;
  int neurons = trainData.numCols()-inputs+1;
  double sensitivity = .05;


  //initializing the weights
  initRand();
  Matrix weights = new Matrix(inputs, neurons);
  weights.rand(-5.0,5.0);
  // weights.print("Initial Weights");
  Matrix activation;

  for(int i = 0; i < iterations; i++){
    //Dot product of weights and inputs
    activation = inputsAndBias.dot(weights);
    //Apply the activation function
    activation.map(activationFunc);
    //Calculate the change in weights (weight -= sensitivity*(input[dot](activation-target))
    //pg 52 in book
    Matrix weightChange = inputsAndBias.transpose();
    Matrix accuracy = activation;
    accuracy.sub(trainDataOutputs);
    weightChange = weightChange.dot(accuracy);
    weightChange.scalarMul(sensitivity);
    weights.sub(weightChange);
  }

  Matrix testOut = outputsAndBias.dot(weights);
  testOut.map(activationFunc);

  Matrix finalOut = new Matrix(testData.numRows(), testData.numCols()+testOut.numCols());
  finalOut.insert(testData, 0, 0);
  finalOut.insert(testOut, 0, testData.numCols());

  finalOut.print();

  return 0;
}

double activationFunc(double x){
  return 1.0/(1.0+exp(-1*actSlope*x));
}
