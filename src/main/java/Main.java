public class Main {

    public static void main(String[] args) {
        Network network = new Network(4, 3, 2, 3);
        double[] output = network.calculate(0.2, 0.3, 0.4, 0.5);
        for (int i = 0; i < 3; i++) {
            System.out.println(output[i]);

        }
    }

}
