public class Network {

    private double[][] output; // Collection of all the outputs of all the neurons. [layers][neurons in layer]
    private final double[][][] weights; //[layers][neurons in layer][connected neuron from previous layer]
    private final double[][] bias; //[layers][neurons]

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

        for (int i = 0; i < numberOfLayers; i++) {
            this.output[i] = new double[layerSizes[i]];
            this.bias[i] = new double[layerSizes[i]];

            if (i > 0) {
                this.weights[i] = new double[layerSizes[i]][layerSizes[i-1]];
            }
        }
    }

    public double[] calculate(double... input) {
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

    private double calculateSigmoidFunction(double x) {
        return 1d / (1 + Math.exp(-x));
    }

}
