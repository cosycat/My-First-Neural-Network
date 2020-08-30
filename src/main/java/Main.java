import java.util.Arrays;
import java.util.Random;

public class Main {

    public static void main(String[] args) {
        int inputSize = 700;
        int outputSize = 700;
        Network network = new Network(inputSize, 100, 100, outputSize);

        double[] inputArray = new double[inputSize];
        Random r = new Random();
        for (int i = 0; i < inputSize; i++) {
            inputArray[i] = r.nextDouble();
        }

        double[] output = network.calculate(inputArray);
        System.out.println(Arrays.toString(output));
    }

}
