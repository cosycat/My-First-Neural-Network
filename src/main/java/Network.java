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
     * Train this NN once with multiple sets of input and expected output.
     * @param trainingCount The number of times this NN shoul be trained with this input/output pairing.
     * @param input The input to calculate the output.
     * @param expectedOutput The output which is to be expected by the input.
     * @param eta The learning rate.
     */
    public void trainMultipleTimes(int trainingCount, double[] input, double[] expectedOutput, double eta) {
        for (int i = 0; i < trainingCount; i++) {
            train(input, expectedOutput, eta);
        }
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
                for (int prevNeuron = 0; prevNeuron < layerSizes[layer - 1]; prevNeuron++) {
                    double outputPrevNeuron = output[layer-1][prevNeuron];
                    double delta = -eta * outputPrevNeuron * errorSignal[layer][neuron];
                    weights[layer][neuron][prevNeuron] += delta;
                }
                double delta = -eta * errorSignal[layer][neuron];
                bias[layer][neuron] += delta;
            }
        }
    }
    
    private double calculateSigmoidFunction(double x) {
        return 1d / (1 + Math.exp(-x));
    }

}
