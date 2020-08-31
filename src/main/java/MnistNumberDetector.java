import mnist.MnistImageFile;
import mnist.MnistLabelFile;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.stream.Stream;

public class MnistNumberDetector {
    
    private final double MAX_PIXEL_BRIGHTNESS = 256.0;
    private final double ETA = 0.3;
    
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
        
        network = new Network(trainingSet.getInputSize(), 16, 16, trainingSet.getOutputSize());
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
    
            int[][] image = imageFile.readImage();
            input = Stream.of(image)
                    .flatMapToInt(Arrays::stream)
                    .mapToDouble(value -> value / MAX_PIXEL_BRIGHTNESS)
                    .toArray();
    
            int label = labelFile.readLabel();
            output[label] = 1.0;
            
            trainingSet.addData(input, output);
            if (dataIndex % (numberOfImages/20) == 0) {
                System.out.println("Read " + dataIndex * 100 / numberOfImages + "% of Data.");
            }
        }
        
        return trainingSet;
    }
    
    
    public void train() {
        network.trainWithDataSet(trainingSet, 1, ETA, true);
    }
    
    public void test() {
        double mseAverage = network.calcualteMSEAverage(testSet);
        System.out.println("MSE of test set: " + mseAverage);
    }
    
    
    
    
    public TrainingSet getTrainingSet() {
        return trainingSet;
    }
}
