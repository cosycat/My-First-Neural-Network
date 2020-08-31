import java.util.ArrayList;
import java.util.Arrays;

public class TrainingSet {
    
    private final int inputSize;
    private final int outputSize;
    
    private final ArrayList<TrainingData> trainingData = new ArrayList<>();
    
    public TrainingSet(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }
    
    public void addData(double[] input, double[] expectedOutput) {
        if (input.length != inputSize || expectedOutput.length != outputSize) throw new IllegalArgumentException("New Data Set has the wrong size.");
        addData(new TrainingData(input, expectedOutput));
    }
    
    public void addData(TrainingData newSet) {
        if (newSet.getInput().length != inputSize || newSet.getExpectedOutput().length != outputSize) throw new IllegalArgumentException("New Data Set has the wrong size.");
        trainingData.add(newSet);
    }
    
    public TrainingData getData(int index) {
        return trainingData.get(index);
    }
    
    @Override
    public String toString() {
        String s = " Training Set [ " + inputSize + " : " + outputSize + " ]\n";
        s +=       "----------------------\n";
        for (int i = 0; i < getDataCount(); i++) {
            s += i + ": " + Arrays.toString(trainingData.get(i).getInput()) + " ->||-> " + Arrays.toString(trainingData.get(i).getExpectedOutput()) + "\n";
        }
        s +=       "----------------------\n";
        return s;
    }
    
    public void printOnConsole() {
        System.out.println(" Training Set [ " + inputSize + " : " + outputSize + " ]\n");
        System.out.println("----------------------\n");
        for (int i = 0; i < getDataCount(); i++) {
            System.out.println(i + ": " + Arrays.toString(trainingData.get(i).getInput()) + " ->||-> " + Arrays.toString(trainingData.get(i).getExpectedOutput()) + "\n");
        }
        System.out.println("----------------------\n");
    }
    
    public int getDataCount() {
        return trainingData.size();
    }
    
    public double[] getInput(int index) {
        return trainingData.get(index).getInput();
    }
    
    public double[] getOutput(int index) {
        return trainingData.get(index).getExpectedOutput();
    }
    
}
