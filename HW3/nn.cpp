//TODO

/*
##make sure input is properly configured
##implement convolution for data
##SEE seriesSampleCol in new matrix library!!!
make sure normalization is configured
    will this change at all?
##remove sigmoid from final layer
##Update respective backprop formula
##"be sure to scale this value by dividing by number of samples(rows) as seen in author's example"
    will have to read more on this
add momentum parameter
check that the training process doesn't assume input==2 or uses any constants
format output
 */


/*
Other stuff:
  deactivate matrix at the end?
 */

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
void updateWeights(Matrix* weights, Matrix inputs, Matrix* activation, Matrix targets, int layers, double sensitivity, Matrix* lastUpdate);
Matrix createLayer(int inputs, int neurons);

static int iterations = 10000;
static double sensitivity = .01;
static double momentum = .1;

int main()
{

  //Reading in the file
  int step;
  scanf("%d", &step);

  int stride;
  scanf("%d", &stride);

  int hiddenNodes;
  scanf("%d", &hiddenNodes);

  Matrix rawData;
  rawData.read();

  //Matrix normData = trainData.normalizeCols();

  //Matrix newData = new Matrix(rows, step+1);
  //newData.insertRowVector(x, extractStride(0, 0, stride, 0));

  Matrix trainData;
  trainData = rawData.seriesSampleCol(0, step, stride);
  Matrix normData = trainData.normalizeCols();
  //2 for this assignment
  int layers = 2;

  //Creating a matrix which separates the inputs and outputs of the training data
  Matrix trainDataInputs = trainData.subMatrix(0, 0, trainData.numRows(), step);
  Matrix trainDataOutputs = trainData.subMatrix(0, step, trainData.numRows(), trainData.numCols()-step);
  
  //Adding a bias
  Matrix trainAndBias = new Matrix(trainDataInputs.numRows(),trainDataInputs.numCols()+1, -1.0);
  trainAndBias.insert(trainDataInputs, 0, 0);
  trainAndBias.print();
  
  //Next we're going to train on the test data (matrix 1)

  //hardcoding the layer values for now
  int nodes[layers+1];
  nodes[0] = step+1;
  nodes[1] = hiddenNodes;
  nodes[2] = 1;

  initRand();
  
  //initializing the weights
  Matrix weights[layers];
  for(int i = 0; i < layers; i++){
    if(i == layers-1)
      weights[i] = createLayer(nodes[i]+1, nodes[i+1]);
    else
      weights[i] = createLayer(nodes[i], nodes[i+1]);
  }

  Matrix activations[2];
  Matrix lastUpdate[2];
  
  for(int i = 0; i < iterations; i++){
    for(int j = 0; j < layers; j++){
    //calculate the outputs for the first layer
      if(j == 0)
        activations[j] = computeLayer(weights[j], trainAndBias, j, layers);
      else
        activations[j] = computeLayer(weights[j], activations[j-1], j, layers);
      //activations[j].print();
    }
    //activations[0].print();
    updateWeights(weights, trainAndBias, activations, trainDataOutputs, layers, sensitivity, lastUpdate);
  }
  //calculating with test data
  activations[0] = computeLayer(weights[0], trainAndBias, 0, layers);
  activations[1] = computeLayer(weights[1], activations[0], 1, layers);

  trainAndBias.insert(activations[1], 0, step);
  
  trainAndBias.unnormalizeCols(normData);
  trainDataOutputs.unnormalizeCols(normData);
  
  Matrix finalOut = new Matrix(trainDataOutputs.numRows(), 2, 0.0);
  finalOut.insert(trainDataOutputs, 0, 0);
  finalOut.insert(trainAndBias.subMatrix(0, step, trainAndBias.numRows(), step-1), 0, 1);
  finalOut.print();

  printf("Sum of distances:\n");
  double dist = trainDataOutputs.dist2(trainAndBias.subMatrix(0, step, trainAndBias.numRows(), step-1));
  printf("%f \n", dist);

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
  if(curLayer < layers-1){
    activation.map(activationFunc);
    Matrix biasActivation = new Matrix(activation.numRows(), activation.numCols()+1, -1.0);
    biasActivation.insert(activation, 0, 0);
    return biasActivation;
  }
  return activation;
}

void updateWeights(Matrix* weights, Matrix inputs, Matrix* activations, Matrix targets, int layers, double sensitivity, Matrix* lastUpdate){

  Matrix gradients[layers];

  gradients[0] = new Matrix(activations[0].numRows(), activations[0].numCols(), 0.0);
  gradients[1] = new Matrix(activations[1].numRows(), activations[1].numCols(), 0.0);


  Matrix weightChange[layers];

  weightChange[0] = new Matrix(weights[0].numRows(), weights[0].numCols(), 0.0);
  weightChange[1] = new Matrix(weights[1].numRows(), weights[1].numCols(), 0.0);
  
  for(int curLayer = layers-1; curLayer >= 0; curLayer--){
    //for the output layer
    if(curLayer == layers-1){
      gradients[curLayer].sub(activations[curLayer]);
      gradients[curLayer].add(targets);
      gradients[curLayer].scalarMul(1.0/((float)inputs.numRows()));
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

  if(lastUpdate[0].isDefined()){
    weights[0].add(lastUpdate[0].scalarMul(momentum));
    weights[1].add(lastUpdate[1].scalarMul(momentum));
  }

  lastUpdate = weightChange;
  //weights[0].print();
  //weights[1].print();

  return;
}

Matrix createLayer(int inputs, int neurons){
  Matrix weights = new Matrix(inputs, neurons, 0.0);
  weights.rand(-1.0,1.0);
  // weights.print("Initial Weights");
  return weights;
}
