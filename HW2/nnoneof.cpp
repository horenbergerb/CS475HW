//TODO

//Implement confusion matrix

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <fstream>
#include "rand.h"
#include "mat.h"

double stepFunc(double x);
double activationFunc(double x);
Matrix computeLayer(Matrix weights, Matrix layerInputs, int curLayer, int layers);
void updateWeights(Matrix* weights, Matrix inputs, Matrix* activation, Matrix targets, int layers, double sensitivity);
Matrix createLayer(int inputs, int neurons);

static int iterations = 10000;
static double sensitivity = .2;

int main()
{

  initRand();

  //Reading in the file
  int inputs;
  scanf("%d", &inputs);
  int hiddenNodes;
  scanf("%d", &hiddenNodes);
  int classes;
  scanf("%d", &classes);

  Matrix trainData;
  trainData.read();
  Matrix testData;
  testData.read();

  Matrix normData = trainData.normalizeCols();
  testData.normalizeCols(normData);

  //2 for this assignment
  int layers = 2;

  //Creating a matrix which separates the inputs and outputs of the training data
  Matrix trainDataInputs = trainData.subMatrix(0, 0, trainData.numRows(), inputs);
  Matrix trainDataOutputs = trainData.subMatrix(0, inputs, trainData.numRows(), trainData.numCols()-inputs);

  //updating classes since the files are not uniformly formatted...
  classes = trainData.numCols()-inputs;

  Matrix testDataInputs = testData.subMatrix(0, 0, testData.numRows(), inputs);
  Matrix testDataOutputs = testData.subMatrix(0, inputs, testData.numRows(), testData.numCols()-inputs);

  //Adding a bias
  Matrix trainAndBias = new Matrix(trainDataInputs.numRows(),trainDataInputs.numCols()+1, -1.0);
  Matrix testAndBias = new Matrix(testDataInputs.numRows(), testDataInputs.numCols()+1, -1.0);
  trainAndBias.insert(trainDataInputs, 0, 0);
  testAndBias.insert(testDataInputs, 0, 0);

  inputs = inputs+1;
  //Next we're going to train on the test data (matrix 1)

  //hardcoding the layer values for now
  int nodes[layers+1];
  nodes[0] = inputs;
  nodes[1] = hiddenNodes;
  nodes[2] = classes;

  //initializing the weights
  Matrix weights[layers];
  for(int i = 0; i < layers; i++){
    if(i == layers-1)
      weights[i] = createLayer(nodes[i]+1, nodes[i+1]);
    else
      weights[i] = createLayer(nodes[i], nodes[i+1]);
  }

  /*  weights[0].print("before");
      weights[1].print("before"); */

  Matrix activations[2];

  for(int i = 0; i < iterations; i++){
    for(int j = 0; j < layers; j++){
    //calculate the outputs for the first layer
      if(j == 0)
        activations[j] = computeLayer(weights[j], trainAndBias, j, layers);
      else
        activations[j] = computeLayer(weights[j], activations[j-1], j, layers);
      //activations[j].print();
    }

    updateWeights(weights, trainAndBias, activations, trainDataOutputs, layers, sensitivity);
  }
  //calculating with test data

  activations[0] = computeLayer(weights[0], testAndBias, 0, layers);
  activations[1] = computeLayer(weights[1], activations[0], 1, layers);

  /*
  weights[0].print("after");
  weights[1].print("after");
  */

  Matrix finalOut = new Matrix(testData.numRows(), testData.numCols()+activations[1].numCols());
  finalOut.insert(testDataInputs, 0, 0);
  finalOut.insert(activations[1].map(stepFunc), 0, testDataInputs.numCols());
  finalOut.insert(testDataOutputs, 0, testDataInputs.numCols()+activations[1].numCols());

  printf("Target\n");
  testDataOutputs.printfmt();
  //finalOut.printfmt();

  Matrix confusion = new Matrix(activations[1].numCols(), activations[1].numCols(), 0.0);
  Matrix output = new Matrix(testDataOutputs.numRows(), 1);

  activations[1].print();

  //Calculating the confusion matrix
  for(int x = 0; x < activations[1].numRows(); x++){
    for(int y = 0; y < activations[1].numCols(); y++){
      if(activations[1].get(x,y) == 1){
        output.set(x, 0, float(y));
        confusion.inc(y, testDataOutputs.get(x,0));
      }
    }
  }
  printf("Predicted\n");
  output.printfmt();
  printf("Confusion Matrix\n");
  confusion.printfmt();

  return 0;
}

double stepFunc(double x){
  if (x>.5)return 1.0;
  else return 0.0;
}

double activationFunc(double x){
  return 1.0/(1.0+exp(-4.0*x));
}

Matrix computeLayer(Matrix weights, Matrix layerInputs, int curLayer, int layers){
  //Dot the weights and inputs
  Matrix activation = layerInputs.dot(weights);
  //Apply the activation function
  activation.map(activationFunc);
  if(curLayer < layers-1){
    Matrix biasActivation = new Matrix(activation.numRows(), activation.numCols()+1, -1.0);
    biasActivation.insert(activation, 0, 0);
    return biasActivation;
  }
  return activation;
}

void updateWeights(Matrix* weights, Matrix inputs, Matrix* activations, Matrix targets, int layers, double sensitivity){

  Matrix gradients[layers];

  gradients[0] = new Matrix(activations[0].numRows(), activations[0].numCols());
  gradients[0].constant(0.0);
  gradients[1] = new Matrix(activations[1].numRows(), activations[1].numCols());
  gradients[1].constant(0.0);

  Matrix weightChange[layers];

  weightChange[0] = new Matrix(weights[0].numRows(), weights[0].numCols());
  weightChange[0].constant(0.0);
  weightChange[1] = new Matrix(weights[1].numRows(), weights[1].numCols());
  weightChange[1].constant(0.0);
  
  for(int curLayer = layers-1; curLayer >= 0; curLayer--){
    //for the output layer
    if(curLayer == layers-1){
      gradients[curLayer].sub(activations[curLayer]);
      gradients[curLayer].add(targets);
      gradients[curLayer].mul(activations[curLayer]);
      activations[curLayer].scalarMul(-1);
      activations[curLayer].scalarAdd(1);
      gradients[curLayer].mul(activations[curLayer]);
    }
    //for the hidden layers
    else{
      gradients[curLayer] = activations[curLayer];
      Matrix nextPiece = activations[curLayer];
      nextPiece.scalarMul(-1);
      nextPiece.scalarAdd(1);
      gradients[curLayer].mul(nextPiece);
      nextPiece = gradients[curLayer+1];
      nextPiece = nextPiece.dot(weights[curLayer+1].transpose());
      gradients[curLayer].mul(nextPiece);
    }
  }

  weightChange[0] = inputs.transpose().dot(gradients[0].subMatrix(0, 0, gradients[0].numRows(), gradients[0].numCols()-1));
  weightChange[0].scalarMul(sensitivity);
  weightChange[1] = activations[0].transpose().dot(gradients[1]);
  weightChange[1].scalarMul(sensitivity);

  weights[0].add(weightChange[0]);
  weights[1].add(weightChange[1]);
  //weights[0].print();
  //weights[1].print();

  return;
}

Matrix createLayer(int inputs, int neurons){
  Matrix weights = new Matrix(inputs, neurons);
  weights.rand(-1.0,1.0);
  // weights.print("Initial Weights");
  return weights;
}
