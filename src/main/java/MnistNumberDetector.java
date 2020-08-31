import mnist.MnistImageFile;
import mnist.MnistLabelFile;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

public class MnistNumberDetector {
    
    private final double MAX_PIXEL_BRIGHTNESS = 256.0;
    private final double ETA = 0.3;
    
    private final List<Integer> HIDDEN_LAYERS = List.of(16, 10);
    
    private final TrainingSet trainingSet;
    private final TrainingSet testSet;
    private final Network network;
    
    public MnistNumberDetector() throws IOException {
        System.out.println("Loading Training Data...");
        this.trainingSet = loadTrainingSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        System.out.println("Training Data Loaded!");

        System.out.println("Loading Test Data...");
        this.testSet = loadTrainingSet("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
        System.out.println("Test Data Loaded!");
    
        ArrayList<Integer> allLayers = new ArrayList<>();
        allLayers.add(trainingSet.getInputSize());
        allLayers.addAll(HIDDEN_LAYERS);
        allLayers.add(trainingSet.getOutputSize());
        network = new Network(allLayers.stream().mapToInt(Integer::intValue).toArray());
        
        train(10);
        calculateMseOfTestSet();
        System.out.println("Error in test set: " + calculateErrorPercentage(testSet) + "%");
        System.out.println("Error in training set: " + calculateErrorPercentage(trainingSet) + "%");
    }
    
    private TrainingSet loadTrainingSet(String imageFileName, String labelFileName) throws IOException {
        
        String path = new File("").getAbsolutePath();
        path += "/src/main/resources/";
        MnistImageFile imageFile = new MnistImageFile(path + imageFileName, "rw");
        MnistLabelFile labelFile = new MnistLabelFile(path + labelFileName, "rw");
        
        int numberOfImages = imageFile.getCount();
        int numberOfLabels = labelFile.getCount();
        assert numberOfImages == numberOfLabels;
    
        TrainingSet trainingSet = new TrainingSet(imageFile.getEntryLength(), 10);
    
        
        for (int dataIndex = 0; dataIndex < numberOfImages; dataIndex++) {
            double[] input = new double[imageFile.getEntryLength()];
            double[] output = new double[10];
    
            for (int pixel = 0; pixel < input.length; pixel++) {
                input[pixel] = imageFile.read();
            }
    
            int label = labelFile.readLabel();
            output[label] = 1.0;
            
            trainingSet.addData(input, output);
            if (dataIndex % (numberOfImages/20) == 0) {
                System.out.println("Read " + dataIndex * 100 / numberOfImages + "% of Data.");
            }
        }
        
        return trainingSet;
    }
    
    
    public void train(int trainingCycles) {
        network.trainWithDataSet(trainingSet, trainingCycles, ETA, true);
    }
    
    public void calculateMseOfTestSet() {
        double mseAverage = network.calcualteMSEAverage(testSet);
        System.out.println("MSE of test set: " + mseAverage);
    }
    
    
    /**
     * Calculates the Error Percentage of all the Test Images in the given TrainingSet.
     *
     * @param set The TrainingSet with test data.
     * @return The Error in %.
     */
    public double calculateErrorPercentage(TrainingSet set) {
        double countCorrect = 0;
        double countWrong = 0;
        for (TrainingData data : set.getTrainingData()) {
            int expectedDigit = ArrayHelperMethods.indexOfHighestValue(data.getExpectedOutput());
            int actualDigit = ArrayHelperMethods.indexOfHighestValue(network.calculateOutput(data.getInput()));
            if (expectedDigit == actualDigit) {
                countCorrect++;
            } else {
                countWrong++;
            }
        }
        
        return countCorrect * 100 / (countCorrect+countWrong);
    }
    
    
    public TrainingSet getTrainingSet() {
        return trainingSet;
    }
}
