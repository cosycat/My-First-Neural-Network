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
    
        System.out.println("Output after     0 trainings:" + Arrays.toString(network.calculateOutput(input)));
    
        network.trainMultipleTimes(10, input, expectedOutput, eta);
        System.out.println("Output after    10 trainings:" + Arrays.toString(network.calculateOutput(input)));
        
        network.trainMultipleTimes(90, input, expectedOutput, eta);
        System.out.println("Output after   100 trainings:" + Arrays.toString(network.calculateOutput(input)));
    
        network.trainMultipleTimes(900, input, expectedOutput, eta);
        System.out.println("Output after  1000 trainings:" + Arrays.toString(network.calculateOutput(input)));
    
        network.trainMultipleTimes(9000, input, expectedOutput, eta);
        System.out.println("Output after 10000 trainings:" + Arrays.toString(network.calculateOutput(input)));
        
        System.out.println("Expected output:" + Arrays.toString(expectedOutput));


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
