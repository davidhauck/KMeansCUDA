//David Hauck
//December 6, 2014

#include <stdio.h>
#include <cassert>

#include <float.h>

#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>

// CUDA helper functions
#include <helper_cuda.h>

#define N 1000000
#define BLOCK_SIZE 32
#define K 4
#define BLOCK_SIZE_CLUSTERS 256

void runKMeansCUDA(int argc, char **argv);
void runKMeansCPU();

//dont need to sqrt, just seeing if a value is larger than others
__device__ double distance2(double x1, double y1, double x2, double y2)
{
	double x = x2 - x1;
	double y = y2 - y1;
	return x*x + y*y;
}

//dont need to sqrt, just seeing if a value is larger than others
double distance2CPU(double x1, double y1, double x2, double y2)
{
	double x = x2 - x1;
	double y = y2 - y1;
	return x*x + y*y;
}

__global__ void calcDistances(double* x, double* y, double* global_xNodes, double* global_yNodes, int* chosenNodes, int* changedNodes)
{
	int id = threadIdx.x;
	int block_offset = blockIdx.x*blockDim.x;

	//shared memory is about 100x faster than global. Its worth it to copy to shared memory
	__shared__ double xs[BLOCK_SIZE];
	__shared__ double ys[BLOCK_SIZE];
	__shared__ double xNodes[K];
	__shared__ double yNodes[K];
	__shared__ int localChangedNodes[BLOCK_SIZE];

	//copying over from global to shared memory goes faster on just one thread per block
	if (threadIdx.x == 0)
	{
		//copy coordinates to shared memory
		for (int i = 0; i < BLOCK_SIZE; i++)
		{
			ys[i] = y[block_offset + i];
			xs[i] = x[block_offset + i];
		}
		//copy cluster centers to shared memory
		for (int i = 0; i < K; i++)
		{
			xNodes[i] = global_xNodes[i];
			yNodes[i] = global_yNodes[i];
		}
	}

	//make sure all memory is copied locally
	__syncthreads();

	//check distances to all cluster nodes. Take the closest one
	double minDistance = distance2(xs[id], ys[id], xNodes[0], yNodes[0]);
	int chosenNode = 0;
	for (int i = 1; i < K; i++)
	{
		double dist = distance2(xs[id], ys[id], xNodes[i], yNodes[i]);
		if (dist < minDistance)
		{
			minDistance = dist;
			chosenNode = i;
		}
	}

	//count how many coordinates changed clusters
	localChangedNodes[id] = 0;
	if (chosenNodes[block_offset + id] != chosenNode)
		localChangedNodes[id] = 1;
	chosenNodes[block_offset + id] = chosenNode;
	__syncthreads();
	for (int i = BLOCK_SIZE / 2; i > 0; i /= 2)
	{
		if (id < i)
		{
			localChangedNodes[id] += localChangedNodes[id + i];
		}
		__syncthreads();
	}
	changedNodes[blockIdx.x] = localChangedNodes[0];
}

__global__ void calcClusterCenters(double* x, double* y, double* global_xNodes, double* global_yNodes, int* chosenNodes)
{
	//each block will do one cluster
	int id = threadIdx.x;
	int k = blockIdx.x;
	__shared__ double xTotal[BLOCK_SIZE_CLUSTERS];
	__shared__ double yTotal[BLOCK_SIZE_CLUSTERS];
	__shared__ int numElements[BLOCK_SIZE_CLUSTERS];
	xTotal[id] = 0.;
	yTotal[id] = 0.;
	numElements[id] = 0.;

	//add up all of the elements belonging to a certain node
	for (int i = id; i < N; i += BLOCK_SIZE_CLUSTERS)
	{
		int chosenNode = chosenNodes[i];
		if (chosenNode == k)
		{
			xTotal[id] += x[i];
			yTotal[id] += y[i];
			numElements[id]++;
		}
	}

	//add together all of the values the nodes in the block found
	__syncthreads();
	for (int i = BLOCK_SIZE_CLUSTERS / 2; i > 0; i /= 2)
	{
		if (id < i)
		{
			xTotal[id] += xTotal[id + i];
			yTotal[id] += yTotal[id + i];
			numElements[id] += numElements[id + i];
		}
		__syncthreads();
	}

	//have one thread update the value with the new cluster center
	if (id == 0)
	{
		double newXPos = -1, newYPos = -1;
		if (numElements[0] != 0)
		{
			newXPos = xTotal[0] / numElements[0];
			newYPos = yTotal[0] / numElements[0];
		}
		global_xNodes[k] = newXPos;
		global_yNodes[k] = newYPos;
	}
}

double my_max(double n1, double n2)
{
	if (n1 > n2)
		return n1;
	return n2;
}

int main(int argc, char **argv)
{
	runKMeansCPU();
	runKMeansCUDA(argc, argv);
	cudaDeviceReset();
	char s[100];
	fgets(s, sizeof(s), stdin);
}

int calcDistancesCPU(double* xCoords, double* yCoords, double* xNodes, double* yNodes, int* chosenNodes)
{
	int changedNodes = 0;
	//for every node, find the closest cluster center
	for (int i = 0; i < N; i++)
	{
		//loop through every cluster center to find the closest
		double minDistance = distance2CPU(xCoords[i], yCoords[i], xNodes[0], yNodes[0]);
		int chosenNode = 0;
		for (int j = 1; j < K; j++)
		{
			double dist = distance2CPU(xCoords[i], yCoords[i], xNodes[j], yNodes[j]);
			if (dist < minDistance)
			{
				minDistance = dist;
				chosenNode = j;
			}
		}

		//check if the node changed clusters
		if (chosenNodes[i] != chosenNode)
			changedNodes++;
		chosenNodes[i] = chosenNode;
	}
	return changedNodes;
}

void calcClusterCentersCPU(double* xCoords, double* yCoords, double* xNodes, double* yNodes, int* chosenNodes)
{
	double* xTotals = (double*)malloc(K * sizeof(double));
	double* yTotals = (double*)malloc(K * sizeof(double));
	int* numElements = (int*)malloc(K * sizeof(int));

	for (int i = 0; i < K; i++)
	{
		xTotals[i] = 0;
		yTotals[i] = 0;
		numElements[i] = 0;
	}

	//add up all the x and y values of all the coordinates for each cluster
	for (int i = 0; i < N; i++)
	{
		//check which cluster the coordinate belongs to
		int chosenNode = chosenNodes[i];
		//add the values to the corresponding cluster
		xTotals[chosenNode] += xCoords[i];
		yTotals[chosenNode] += yCoords[i];
		numElements[chosenNode]++;
	}

	//recalculate the centers
	for (int i = 0; i < K; i++)
	{
		//get the total sums of all the x and y coordinates
		double xTotal = xTotals[i];
		double yTotal = yTotals[i];
		//get the total number of elements
		int numE = numElements[i];
		if (numE > 0)
		{
			//find the average x and y coordinate
			double newX = xTotal / numE;
			double newY = yTotal / numE;
			//update the cluster center with its new value
			xNodes[i] = newX;
			yNodes[i] = newY;
		}
		else
		{
			xNodes[i] = -1;
			yNodes[i] = -1;
		}
	}
	free(xTotals);
	free(yTotals);
}

void runKMeansCPU()
{
	float solveTotal = 0, total = 0;
	for (int j = 0; j < 3; j++)
	{
		printf("CPU iteration %d:\r\n", j);
		cudaEvent_t t1, t2, t3;
		cudaEventCreate(&t1);
		cudaEventCreate(&t2);
		cudaEventCreate(&t3);


		cudaEventRecord(t1, 0);
		cudaEventSynchronize(t1);

		//initialize the data with random values
		double* xCoords = (double*)malloc(N * sizeof(double));
		double* yCoords = (double*)malloc(N * sizeof(double));
		for (int i = 0; i < N; i++)
		{
			xCoords[i] = rand() % 100;
			yCoords[i] = rand() % 100;
		}

		double* yNodes = (double*)malloc(K * sizeof(double));
		double* xNodes = (double*)malloc(K * sizeof(double));
		for (int i = 0; i < K; i++)
		{
			xNodes[i] = rand() % 100;
			yNodes[i] = rand() % 100;
		}

		int* chosenNodes = (int*)malloc(N * sizeof(int));
		for (int i = 0; i < N; i++)
		{
			chosenNodes[i] = -1;
		}

		cudaEventRecord(t2, 0);
		cudaEventSynchronize(t2);

		//keep iterating until less that 1% of the coordinates change clusters
		int changedNodes;
		bool shouldContinue;
		do
		{
			changedNodes = calcDistancesCPU(xCoords, yCoords, xNodes, yNodes, chosenNodes);
			calcClusterCentersCPU(xCoords, yCoords, xNodes, yNodes, chosenNodes);
			//printf("%d\r\n", changedNodes);
			shouldContinue = false;

			//keep looping until all clusters have at least one node (Not doing this in cuda do to memory copying overhead. When timing the runs, this is taken out.)
			/*for (int i = 0; i < K; i++)
			{
				if (xNodes[i] == -1)
				{
					shouldContinue = true;
					xNodes[i] = rand() % (int)maxX;
					yNodes[i] = rand() % (int)maxY;
				}
			}*/
			int x = 0;
		} while (changedNodes > 0.01 * N || shouldContinue);

		//free(xCoords);
		//free(yCoords);
		free(chosenNodes);

		cudaEventRecord(t3, 0);
		cudaEventSynchronize(t3);

		float timeCreate, timeSolve, timeTotal;
		cudaEventElapsedTime(&timeCreate, t1, t2);
		cudaEventElapsedTime(&timeSolve, t2, t3);
		cudaEventElapsedTime(&timeTotal, t1, t3);
		printf("Create Time:%3.1f\r\nSolve Time:%3.1f\r\nTotal Time:%3.1f\r\n", timeCreate, timeSolve, timeTotal);
		printf("Cluster Centers:\r\n");
		for (int i = 0; i < K; i++)
		{
			printf("(%3.1f,\t%3.1f)\r\n", xNodes[i], yNodes[i]);
		}
		solveTotal += timeSolve;
		total += timeTotal;
		free(xNodes);
		free(yNodes);

	}
	float totalTimeAvg = total / 10;
	float solveTimeAvg = solveTotal / 10;
	printf("CPU Solve Time Avg:%3.1f\r\nTotal Time Avg:%3.1f\r\n", solveTimeAvg, totalTimeAvg);
}

void runKMeansCUDA(int argc, char **argv)
{
	float solveTotal = 0, total = 0;
	for (int j = 0; j < 3; j++)
	{
		printf("GPU iteration %d:\r\n", j);
		cudaEvent_t t1, t2, t3, t4, t5;
		cudaEventCreate(&t1);
		cudaEventCreate(&t2);
		cudaEventCreate(&t3);
		cudaEventCreate(&t4);
		cudaEventCreate(&t5);

		cudaEventRecord(t1, 0);
		cudaEventSynchronize(t1);
		//calculate the number of blocks based on the size of N. the block size is fixed at 32
		int Nblocks = N / 32;
		if (N % BLOCK_SIZE != 0)
		{
			Nblocks++;
		}
		int Nthreads = BLOCK_SIZE;

		//initialize the data with random values
		double* xCoords = (double*)malloc(N * sizeof(double));
		double* yCoords = (double*)malloc(N * sizeof(double));
		for (int i = 0; i < N; i++)
		{
			xCoords[i] = rand() % 100;
			yCoords[i] = rand() % 100;
		}

		double* yNodes = (double*)malloc(K * sizeof(double));
		double* xNodes = (double*)malloc(K * sizeof(double));
		for (int i = 0; i < K; i++)
		{
			xNodes[i] = rand() % 100;
			yNodes[i] = rand() % 100;
		}

		int* chosenNodes = (int*)malloc(N * sizeof(int));
		chosenNodes[0] = 16;
		for (int i = 1; i < N; i++)
		{
			chosenNodes[i] = -1;
		}

		int* changedNodes = (int*)malloc(Nblocks * sizeof(int));
		for (int i = 0; i < Nblocks; i++)
		{
			changedNodes[i] = -1;
		}

		cudaEventRecord(t2, 0);
		cudaEventSynchronize(t2);

		int devID;

		cudaError_t error;
		cudaDeviceProp deviceProp;

		devID = findCudaDevice(argc, (const char **)argv);

		checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

		if (deviceProp.major < 2)
		{
			cudaDeviceReset();
			exit(EXIT_SUCCESS);
		}

		//create the matching data on the gpu
		double* d_xNodes;
		checkCudaErrors(cudaMalloc((void **)&d_xNodes, K * sizeof(double)));
		double* d_yNodes;
		checkCudaErrors(cudaMalloc((void **)&d_yNodes, K * sizeof(double)));

		double* d_xCoords;
		checkCudaErrors(cudaMalloc((void **)&d_xCoords, N * sizeof(double)));
		double* d_yCoords;
		checkCudaErrors(cudaMalloc((void **)&d_yCoords, N * sizeof(double)));

		int* d_chosenNodes;
		checkCudaErrors(cudaMalloc((void **)&d_chosenNodes, N * sizeof(int)));

		int* d_changedNodes;
		checkCudaErrors(cudaMalloc((void **)&d_changedNodes, Nblocks * sizeof(int)));

		//send the data to the gpu
		checkCudaErrors(cudaMemcpy(d_xNodes, xNodes, K * sizeof(double), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_yNodes, yNodes, K * sizeof(double), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_xCoords, xCoords, N * sizeof(double), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_yCoords, yCoords, N * sizeof(double), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_chosenNodes, chosenNodes, N * sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_changedNodes, changedNodes, Nblocks * sizeof(int), cudaMemcpyHostToDevice));


		// Kernel configuration, where a one-dimensional
		// grid and one-dimensional blocks are configured.
		dim3 dimGrid(Nblocks);
		dim3 dimBlock(Nthreads);

		dim3 kDim3(K);
		dim3 bscDim3(BLOCK_SIZE_CLUSTERS);

		cudaEventRecord(t3, 0);
		cudaEventSynchronize(t3);

		//keep iterating until less than 1% of the data changes clusters
		int totalChanges;
		do
		{
			calcDistances <<<dimGrid, dimBlock>>>(d_xCoords, d_yCoords, d_xNodes, d_yNodes, d_chosenNodes, d_changedNodes);
			checkCudaErrors(cudaMemcpy(changedNodes, d_changedNodes, Nblocks * sizeof(int), cudaMemcpyDeviceToHost));
			totalChanges = 0;
			for (int i = 0; i < Nblocks; i++)
			{
				totalChanges += changedNodes[i];
			}
			calcClusterCenters <<<kDim3, bscDim3 >>>(d_xCoords, d_yCoords, d_xNodes, d_yNodes, d_chosenNodes);
			printf("%d\r\n", totalChanges);
		} while (totalChanges > 0.01 * N);

		cudaEventRecord(t4, 0);
		cudaEventSynchronize(t4);

		double* newyNodes = (double*)malloc(K * sizeof(double));
		double* newxNodes = (double*)malloc(K * sizeof(double));
		checkCudaErrors(cudaMemcpy(newxNodes, d_xNodes, K * sizeof(double), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(newyNodes, d_yNodes, K * sizeof(double), cudaMemcpyDeviceToHost));
		cudaFree(d_changedNodes);
		cudaFree(d_chosenNodes);
		cudaFree(d_xCoords);
		cudaFree(d_xNodes);
		cudaFree(d_yCoords);
		cudaFree(d_yNodes);
		//free(xCoords);
		//free(yCoords);
		free(xNodes);
		free(yNodes);
		free(chosenNodes);
		free(changedNodes);
		cudaEventRecord(t5, 0);
		cudaEventSynchronize(t5);

		float timeCreate, timeSetup, timeSolve, timeFree, timeTotal;
		cudaEventElapsedTime(&timeCreate, t1, t2);
		cudaEventElapsedTime(&timeSetup, t2, t3);
		cudaEventElapsedTime(&timeSolve, t3, t4);
		cudaEventElapsedTime(&timeFree, t4, t5);
		cudaEventElapsedTime(&timeTotal, t1, t5);
		printf("Create Time:%3.1f\r\nCuda Setup Time:%3.1f\r\nSolve Time:%3.1f\r\nDownload Answer and Free Memory Time:%3.1f\r\nTotal Time:%3.1f\r\n", timeCreate, timeSetup, timeSolve, timeFree, timeTotal);
		printf("Cluster Centers:\r\n");

		for (int i = 0; i < K; i++)
		{
			printf("(%3.1f,\t%3.1f)\r\n", newxNodes[i], newyNodes[i]);
		}
		solveTotal += timeSolve;
		total += timeTotal;
	}
	//free(xCoords);
	//free(yCoords);
	float totalTimeAvg = total / 10;
	float solveTimeAvg = solveTotal / 10;
	printf("GPU Solve Time Avg:%3.1f\r\nTotal Time Avg:%3.1f\r\n", solveTimeAvg, totalTimeAvg);
}
