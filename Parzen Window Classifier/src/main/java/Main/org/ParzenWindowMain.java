package Main.org;

import Data.org.DataSource;
import Data.org.Spliterator;
import ParzenWindowClassifier.org.ParzenWindowClassifier;

/**
 * This is the main class in which we apply the Parzen window with the bayesian decision to classify the features vectors
 */
public class ParzenWindowMain {
    public static void main(String[] args) {
        // Read the .csv file of the data
        DataSource dataSource = new DataSource("...\\data.csv", "class");
        var dataset = dataSource.readData();

        // Split the data into two parts, training and testing
        var split = new Spliterator(dataset, 0.5f, 12345);
        var train = split.getTrainData();
        var test = split.getTestData();

        // Creat an object from the Parzen Window classifier
        ParzenWindowClassifier parzenWindowClassifier = new ParzenWindowClassifier(2);
        // Train the model. Note, this is a lazy algorithm
        parzenWindowClassifier.train(train);
        // Test the model
        parzenWindowClassifier.test(test);
        // Print out the accuracy
        System.out.println("The classification accuracy : " + parzenWindowClassifier.getAcc());
    }
}
