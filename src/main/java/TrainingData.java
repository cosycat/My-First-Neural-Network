public class TrainingData {
    
    private final double[] input;
    private final double[] expectedOutput;
    
    public TrainingData(double[] input, double[] expectedOutput) {
        this.input = input;
        this.expectedOutput = expectedOutput;
    }
    
    public double[] getInput() {
        return input;
    }
    
    public double[] getExpectedOutput() {
        return expectedOutput;
    }
}
