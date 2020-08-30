import java.util.Arrays;
import java.util.Random;

public class Main {

    public static void main(String[] args) {
        int inputSize = 5;
        int outputSize = 3;
        Network network = new Network(inputSize, 2, 3, outputSize);
        
        double[] input = {0.3, 0.5, 0.1, 0.8, 0.5};
        double[] expectedOutput = {0, 1, 1};
        double eta = 0.3;
        
        network.trainWithExponentialOutput(3, input, expectedOutput, eta);
   


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
