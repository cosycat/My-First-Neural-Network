import mnist.MnistImageFile;
import mnist.MnistLabelFile;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.stream.Stream;

public class MnistNumberDetector {
    
    private final double MAX_PIXEL_BRIGHTNESS = 256.0;
    
    private final TrainingSet trainingSet;
    
    public MnistNumberDetector() throws IOException {
        this.trainingSet = loadTrainingSet();
        System.out.println("MnistNumberDetector.MnistNumberDetector - Data Loaded!");
        System.out.println(trainingSet);
    }
    
    private TrainingSet loadTrainingSet() throws IOException {
        
        String path = new File("").getAbsolutePath();
        path += "/src/main/resources";
        MnistImageFile imageFile = new MnistImageFile(path + "/train-images.idx3-ubyte", "rw");
        MnistLabelFile labelFile = new MnistLabelFile(path + "/train-labels.idx1-ubyte", "rw");
        
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
    
    public TrainingSet getTrainingSet() {
        return trainingSet;
    }
}
