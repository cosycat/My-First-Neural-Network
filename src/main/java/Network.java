import java.util.Arrays;

public class Network {

    private double[][] output; // Collection of all the outputs of all the neurons. [layers][neurons in layer]
    private final double[][][] weights; //[layers][neurons in layer][connected neuron from previous layer]
    private final double[][] bias; //[layers][neurons]

    private final double[][] errorSignal; //[layers][neurons]

    private final int numberOfLayers;
    private final int[] layerSizes;
    private final int inputLayerSize;
    private final int outputLayerSize;

    public Network(int... layerSizes) {
        this.numberOfLayers = layerSizes.length;
        this.layerSizes = layerSizes;
        this.inputLayerSize = layerSizes[0];
        this.outputLayerSize = layerSizes[numberOfLayers - 1];

        this.output = new double[numberOfLayers][];
        this.weights = new double[numberOfLayers][][];
        this.bias = new double[numberOfLayers][];

        this.errorSignal = new double[numberOfLayers][];

        for (int i = 0; i < numberOfLayers; i++) {
            this.output[i] = new double[layerSizes[i]];
            this.errorSignal[i] = new double[layerSizes[i]];
            this.bias[i] = ArrayHelperMethods.createRandomArray(layerSizes[i], 0.3, 0.7);

            if (i > 0) {
                this.weights[i] = ArrayHelperMethods.createRandom2DArray(layerSizes[i], layerSizes[i-1], -0.3, 0.5);
            }
        }
    }

    public double[] calculateOutput(double... input) {
        if (input.length != inputLayerSize) throw new IllegalArgumentException("The size of the input did not match the size of the input layers");
        this.output[0] = input;
        for (int layer = 1; layer < numberOfLayers; layer++) {
            for (int neuron = 0; neuron < layerSizes[layer]; neuron++) {

                double sum = 0;
                for (int prevNeuron = 0; prevNeuron < output[layer - 1].length; prevNeuron++) {
                    sum += weights[layer][neuron][prevNeuron] * output[layer - 1][prevNeuron];
                }
                sum += bias[layer][neuron];

                output[layer][neuron] = calculateSigmoidFunction(sum);
            }
        }
        return output[numberOfLayers - 1];
    }

    /**
     * Train this NN once with one set of input and expected output.
     * @param input The input to calculate the output.
     * @param expectedOutput The output which is to be expected by the input.
     * @param eta The learning rate.
     */
    public void train(double[] input, double[] expectedOutput, double eta) {
        if (input.length != inputLayerSize || expectedOutput.length != outputLayerSize) throw new IllegalArgumentException("The sizes of the input or expected output weren't matching the size of the Network.");
        calculateOutput(input);
        backpropError(expectedOutput);
        updateWeightsAndBias(eta);
    }
    
    /**
     * Train this NN once with every entry of the given TrainingSet.
     *
     * @param set The given TrainingSet.
     * @param eta The learning rate.
     * @param printLog Whether the MSE should be calculated and printed.
     */
    public void trainWithDataSet(TrainingSet set, double eta, boolean printLog) {
        trainWithDataSet(set, 1, eta, printLog);
    }
    
    /**
     * Train this NN multiple times with every entry of the given TrainingSet.
     *
     * @param set The given TrainingSet.
     * @param trainingCycles How many times the whole set should be trained with.
     * @param eta The learning rate.
     * @param printLog Whether the MSE should be calculated and printed.
     */
    public void trainWithDataSet(TrainingSet set, int trainingCycles, double eta, boolean printLog) {
        for (int cycle = 0; cycle < trainingCycles; cycle++) {
            for (int trainingData = 0; trainingData < set.getDataCount(); trainingData++) {
                train(set.getInput(trainingData), set.getOutput(trainingData), eta);
                if (printLog && trainingData % (set.getDataCount()/20) == 0) {
                    System.out.println("Trained with " + trainingData * 100 / set.getDataCount() + "% of Data.");
                }
            }
            System.out.println("Network.trainWithDataSet - Cycle " + (cycle + 1) + " completed!");
        }
        if (printLog) {
            double mse = calcualteMSEAverage(set);
            System.out.println("MSE after " + trainingCycles + " training cycles: " + mse);
        }
    }
    
    public void trainWithDataSet(TrainingSet set, int trainingCycles, double eta) {
        trainWithDataSet(set, trainingCycles, eta, true);
    }
    

    private void backpropError(double[] expectedOutputs) {
        for (int neuron = 0; neuron < outputLayerSize; neuron++) {
            double actualOutput = output[numberOfLayers - 1][neuron];
            double expectedOutput = expectedOutputs[neuron];
            double outputDerivative = actualOutput * (1 - actualOutput);
            errorSignal[numberOfLayers - 1][neuron] = (actualOutput - expectedOutput) * outputDerivative;
        }

        for (int layer = numberOfLayers - 2; layer > 0; layer--) {
            for (int neuron = 0; neuron < layerSizes[layer]; neuron++) {
                double sum = 0;
                for (int nextNeuron = 0; nextNeuron < layerSizes[layer + 1]; nextNeuron++) {
                    sum += weights[layer + 1][nextNeuron][neuron] * errorSignal[layer + 1][nextNeuron];
                }
                double derivative = output[layer][neuron] * (1 - output[layer][neuron]);
                errorSignal[layer][neuron] = sum * derivative;
            }
        }
    }

    private void updateWeightsAndBias(double eta) {
        for (int layer = 1; layer < numberOfLayers; layer++) {
            for (int neuron = 0; neuron < layerSizes[layer]; neuron++) {
                
                // Update bias
                double delta = -eta * errorSignal[layer][neuron];
                bias[layer][neuron] += delta;
                
                // Update weights
                for (int prevNeuron = 0; prevNeuron < layerSizes[layer - 1]; prevNeuron++) {
                    double outputPrevNeuron = output[layer-1][prevNeuron];
                    delta *= outputPrevNeuron;
                    weights[layer][neuron][prevNeuron] += delta;
                }
                
            }
        }
    }
    
    public double calculateMSE(TrainingData trainingData) {
        return calculateMSE(trainingData.getInput(), trainingData.getExpectedOutput());
    }
    
    public double calculateMSE(double[] input, double[] expectedOutput) {
        if (input.length != inputLayerSize || expectedOutput.length != outputLayerSize) throw new IllegalArgumentException("calculateMSE failed because of wrong Data size.");
        calculateOutput(input);
        double v = 0;
        for (int i = 0; i < expectedOutput.length; i++) {
            v += Math.pow(expectedOutput[i] - output[numberOfLayers - 1][i], 2);
        }
        return v / (2d * expectedOutput.length);
    }
    
    public double calcualteMSEAverage(TrainingSet set) {
        double v = 0;
        for (int i = 0; i < set.getDataCount(); i++) {
            v += calculateMSE(set.getData(i));
        }
        return v / set.getDataCount();
    }
    
    
    private double calculateSigmoidFunction(double x) {
        return 1d / (1 + Math.exp(-x));
    }
    
}
