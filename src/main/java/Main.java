import java.io.IOException;

public class Main {
    
    
    public static void main(String[] args) throws IOException {
    
        MnistNumberDetector numberDetector = new MnistNumberDetector();
        
    }
    
    
    
    
    
    
    
    
    
    

    /*public static void main(String[] args) {
        int inputSize = 28 * 28;
        int outputSize = 10;
        Network network = new Network(inputSize, 3, 3, outputSize);

        double eta = 0.3;
        
        double[] input = {0.6, 0.1, 0.4, 0.8};
        double[] expectedOutput = {0.1, 0.9};
        
        double[] input2 = {0.1, 0.2, 0.3, 0.4};
        double[] expectedOutput2 = {0.9, 0.1};
        
        TrainingSet trainingSet = new TrainingSet(inputSize, outputSize);
        trainingSet.addData(input, expectedOutput);
        trainingSet.addData(input2, expectedOutput2);
        
        network.trainWithDataSet(trainingSet, 10000, 0.3, true);
   


        /*double[] inputArray = new double[inputSize];
        Random r = new Random();
        for (int i = 0; i < inputSize; i++) {
            inputArray[i] = r.nextDouble();
        }
        double[] output = network.calculateOutput(inputArray);
        System.out.println(Arrays.toString(output));
        *//*
    }*/

}
