import java.util.Arrays;
import java.util.Random;

public class Main {

    public static void main(String[] args) {
        int inputSize = 4;
        int outputSize = 2;
        Network network = new Network(inputSize, 3, 3, outputSize);

        double eta = 0.3;
        
        double[] input = {0.6, 0.1, 0.4, 0.8};
        double[] expectedOutput = {0.1, 0.9};
        
        double[] input2 = {0.1, 0.2, 0.3, 0.4};
        double[] expectedOutput2 = {0.9, 0.1};
        
        TrainingSet trainingSet = new TrainingSet(inputSize, outputSize);
        trainingSet.addData(input, expectedOutput);
        trainingSet.addData(input2, expectedOutput2);
        
        network.trainWithDataSet(trainingSet, 10000, 0.3);
    
        System.out.println(trainingSet);
    
        System.out.println(Arrays.toString(network.calculateOutput(input)));
        System.out.println(Arrays.toString(network.calculateOutput(input2)));
   


        /*double[] inputArray = new double[inputSize];
        Random r = new Random();
        for (int i = 0; i < inputSize; i++) {
            inputArray[i] = r.nextDouble();
        }
        double[] output = network.calculateOutput(inputArray);
        System.out.println(Arrays.toString(output));
        */
    }

}
