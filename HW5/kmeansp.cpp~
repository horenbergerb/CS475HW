#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "rand.h"
#include "mat.h"

double randCoord(double min, double max);
Matrix getRow(Matrix curMatrix, int row);

int main(int argc, char* argv[]){
  //takes two inputs:
  //k: Number of expect groups
  //t: number of times kmeans runs
  int k;
  int t;

  srand(time(NULL));
  
  if(argc == 3){
    k = atoi(argv[1]);
    t = atoi(argv[2]);
  }
  else{
    printf("You messed up\n");
    exit(0);
  }

  //Loading our data
  Matrix rawData;
  rawData.read();

  //Kmeans initializes K points with each dimension randomly between the min and max
  //KmeansP will pick random points in the data to be our centers

  Matrix centers(k, rawData.numCols(), 0.0);
  
  for(int column = 0; column < rawData.numCols(); column++){
    for(int cur_center = 0; cur_center < k; cur_center++){
      double next_value = randCoord(rawData.minCol(column), rawData.maxCol(column));
      centers.set(cur_center, column, next_value);
    }
  }
  
  //Run the K-means algorithm to find the k center points
  for(int loops = 0; loops < t; loops++){
    //centers.print();
    //First we calculate the distances of all points to all means
    Matrix distances(rawData.numRows(), k , 0.0);
    for(int cur_point = 0; cur_point < rawData.numRows(); cur_point++){
      for(int cur_center = 0; cur_center < k; cur_center++){
	Matrix row1 = getRow(rawData, cur_point);
	Matrix row2 = getRow(centers, cur_center);
	float cur_dist = row1.dist2(row2);
	//Contains pointXcenter matrix of distances 
	distances.set(cur_point, cur_center, cur_dist);
      }
    }
    //distances.print();
    //Next, we calculate the new means
    //Our new centers
    Matrix newCenters(k, rawData.numCols(), 0.0);
    int was_broke = 0;
    //For each center, we count the points which are nearest, add those values and take the average
    for(int cur_center = 0; cur_center < k; cur_center++){
      double point_count = 0.0;
      //for all of the points, if the distance is smallest, add the point to the mean
      for(int cur_point = 0; cur_point < rawData.numRows(); cur_point++){
	int min_row;
	int min_col;
	//getRow(distances, cur_point).print();
	getRow(distances, cur_point).argMin(min_row, min_col);
	//printf("%i nearest center\n", min_col);
	if(min_col == cur_center){
	  newCenters = newCenters.addRowVector(cur_center, getRow(rawData, cur_point));
	  point_count += 1.0;
	}
      }
      //Catching cases where some centers never had any points
      if (point_count == 0.0){
	//printf("No mean at %i\n", cur_center);
	double next_value0 = randCoord(rawData.minCol(0), rawData.maxCol(0));
	double next_value1 = randCoord(rawData.minCol(1), rawData.maxCol(1));
	centers.set(cur_center, 0, next_value0);
	centers.set(cur_center, 1, next_value1);
	loops = loops-1;
	was_broke = 1;
	break;
      }
      for(int col = 0; col < newCenters.numCols(); col++){
	newCenters.set(cur_center, col, newCenters.get(cur_center, col)*(1.0/point_count));
      }
    }
    if(!was_broke){
      centers = newCenters;
      centers = centers.sortRows();
      centers.printfmt("Points:");
      Matrix center_dist(k, k, 0.0);
      double min_dist = 100000000.0;
      for(int dist1 = 0; dist1 < k; dist1++){
	for(int dist2 = dist1+1; dist2 < k; dist2++){
	  if(getRow(centers, dist1).dist2(getRow(centers, dist2)) < min_dist)
	    min_dist = getRow(centers, dist1).dist2(getRow(centers, dist2));
	}
      }
  printf("K: %i MinD: %f\n", k, min_dist);

    }
  }
  
}

double randCoord(double min, double max) {
    double random = ((double) rand()) / (double) RAND_MAX;
    double diff = max - min;
    double increment = random * diff;

    return min + increment;
}

Matrix getRow(Matrix curMatrix, int row){
  Matrix rowMatrix = curMatrix.extract(row, 0, 1, curMatrix.numCols());
  return rowMatrix;
}
