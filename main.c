#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "mnist.h"

void shuffle(double *array, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            double t = array[j];
            array[j] = array[i];
            array[i] = t;

            j = i + rand() / (RAND_MAX / (n - i) + 1);
            t = array[j];
            array[j] = array[i];
            array[i] = t;

        }
    }
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double dSigmoid(double x)
{
    return x * (1.0 - x);
}

double initWeight()
{
    return (double)rand() / RAND_MAX;
}


#define INPUTS 784
#define HIDDEN_LAYERS 2
#define HIDDEN_NODES 4
#define OUTPUTS 10

#define SIZE 785 // 28*28 (+1 for label)
#define NUM_TRAIN 60000

double trainingSet[NUM_TRAIN][SIZE];

int trainingSetSize = NUM_TRAIN;

double input[INPUTS];

double hidden[HIDDEN_LAYERS][HIDDEN_NODES];
double hiddenWeights[HIDDEN_LAYERS][HIDDEN_NODES][INPUTS];

double output[OUTPUTS];
double outputWeights[OUTPUTS][HIDDEN_NODES];

double hiddenBias[HIDDEN_LAYERS][HIDDEN_NODES];
double outputBias[OUTPUTS];

double hiddenError[HIDDEN_LAYERS][HIDDEN_NODES];
double outputError[OUTPUTS];

void initNetwork()
{
    for(int i = 0; i < HIDDEN_LAYERS; i++)
    {
        for(int j = 0; j < HIDDEN_NODES; j++)
        {
            hiddenBias[i][j] = initWeight();
            for(int k = 0; k < INPUTS; k++)
            {
                hiddenWeights[i][j][k] = initWeight();
            }
        }
    }

    for(int i = 0; i < OUTPUTS; i++)
    {
        outputBias[i] = initWeight();
        for(int j = 0; j < HIDDEN_NODES; j++)
        {
            outputWeights[i][j] = initWeight();
        }
    }
}


void feedForward()
{
    for(int i = 0; i < HIDDEN_LAYERS; i++)
    {
        for(int j = 0; j < HIDDEN_NODES; j++)
        {
            double sum = 0;
            for(int k = 0; k < INPUTS; k++)
            {
                sum += input[k] * hiddenWeights[i][j][k];
            }
            hidden[i][j] = sigmoid(sum + hiddenBias[i][j]);
        }
    }

    for(int i = 0; i < OUTPUTS; i++)
    {
        double sum = 0;
        for(int j = 0; j < HIDDEN_NODES; j++)
        {
            sum += hidden[HIDDEN_LAYERS - 1][j] * outputWeights[i][j];
        }
        output[i] = sigmoid(sum + outputBias[i]);
    }
}

void backPropagate(double target)
{
    for(int i = 0; i < OUTPUTS; i++)
    {
        outputError[i] = dSigmoid(output[i]) * (target - output[i]);
    }

    for(int i = 0; i < HIDDEN_LAYERS; i++)
    {
        for(int j = 0; j < HIDDEN_NODES; j++)
        {
            double sum = 0;
            for(int k = 0; k < OUTPUTS; k++)
            {
                sum += outputError[k] * outputWeights[k][j];
            }
            hiddenError[i][j] = dSigmoid(hidden[i][j]) * sum;
        }
    }
}

void updateWeights(double learningRate)
{
    for(int i = 0; i < OUTPUTS; i++)
    {
        for(int j = 0; j < HIDDEN_NODES; j++)
        {
            outputWeights[i][j] += learningRate * outputError[i] * hidden[HIDDEN_LAYERS - 1][j];
        }
        outputBias[i] += learningRate * outputError[i];
    }

    for(int i = 0; i < HIDDEN_LAYERS; i++)
    {
        for(int j = 0; j < HIDDEN_NODES; j++)
        {
            for(int k = 0; k < INPUTS; k++)
            {
                hiddenWeights[i][j][k] += learningRate * hiddenError[i][j] * input[k];
            }
            hiddenBias[i][j] += learningRate * hiddenError[i][j];
        }
    }
}

void trainNetwork(double learningRate, int epochs)
{
    for(int i = 0; i < epochs; i++)
    {
        for(int j = 0; j < trainingSetSize; j++)
        {
            shuffle(&trainingSet, trainingSetSize);
            for(int k = 0; k < INPUTS; k++)
            {
                input[k] = trainingSet[j][k];
            }
            feedForward();
            backPropagate(trainingSet[j][INPUTS]);
            updateWeights(learningRate);

            printf("Epoch: %d, Training: %d, Error: %f, max: ", i, j, outputError[0]);
            double max = 0;
            int maxIndex = 0;
            for (int k = 0; k < 10; ++k) {
                if (output[k] > max) {
                    max = output[k];
                    maxIndex = k;
                }
            }
            printf("%d, prob: %f \n", maxIndex, max);

        }
    }


}

void testNetwork()
{
    for(int i = 0; i < trainingSetSize; i++)
    {
        for(int j = 0; j < INPUTS; j++)
        {
            input[j] = trainingSet[i][j];
        }

        feedForward();

        for(int j = 0; j < OUTPUTS; j++)
        {
            printf("%f ", output[j]);
        }
    }
}

void readData() {
    load_mnist();
    for(int i = 0; i < NUM_TRAIN; i++)
    {
        for(int j = 0; j < INPUTS; j++)
        {
            trainingSet[i][j] = (double)train_image[i][j] / 255.0;
        }
        trainingSet[i][INPUTS] = (double)train_label[i] / 255.0;
    }
}


int main () {
    readData();
    initNetwork();
    trainNetwork(0.1, 10000);

    for (int i = 0; i < 5; i++) {
        printf("\n\n\n");
    }

    testNetwork();
    printf("\n");

    return 0;
}