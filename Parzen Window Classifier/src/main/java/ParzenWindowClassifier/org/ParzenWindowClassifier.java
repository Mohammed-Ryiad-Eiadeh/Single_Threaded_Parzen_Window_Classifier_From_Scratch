package ParzenWindowClassifier.org;

import Data.org.DataSource;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

/**
 * This class is defined a parzen window classification algorithm
 */
public class ParzenWindowClassifier {
    // Bandwidth parameter for Parzen Window
    private final int bandWidth;
    // Mapping from class IDs to their respective samples
    private final HashMap<String, double[][]> mapClassToSamples;
    // Mapping from class IDs to their priors
    private final HashMap<String, Float> mapClassToPriors;
    // Accuracy counter
    private double acc;

    /**
     * Constructor to initialize the classifier with a bandwidth
     * @param bandWidth The bandwidth which is the window
     */
    public ParzenWindowClassifier(int bandWidth) {
        this.bandWidth = bandWidth;
        mapClassToSamples = new HashMap<>();
        mapClassToPriors = new HashMap<>();
        this.acc = 0.0;
    }

    /**
     * This method is used to train the model
     * @param train The training part of the data
     */
    // Method to train the classifier with labeled samples
    public void train(ArrayList<DataSource.sample_Label> train) {
        // Fetch the unique class IDs present in the training data
        var classesId = train.stream().map(DataSource.sample_Label::label).distinct().toList();

        // Compute the number of samples per class and their priors
        for (String currentClass : classesId) {
            // Filter and collect samples corresponding to the current class
            var samplesPerClass = train.stream()
                    .filter(vec -> Double.parseDouble(vec.label()) == Double.parseDouble(currentClass))
                    .map(DataSource.sample_Label::sample)
                    .toList()
                    .toArray(double[][]::new);

            // Store samples and their priors in the corresponding maps
            mapClassToSamples.put(currentClass, samplesPerClass);
            mapClassToPriors.put(currentClass, (float) samplesPerClass.length / train.size());
        }
    }

    /**
     * This Method is used to compute the density for test samples and return accuracy
     * @param test The array list of samples corresponding the testing phase
     * @return The classification accuracy
     */
    public void test(ArrayList<DataSource.sample_Label> test) {
        // Iterate over test samples
        for (DataSource.sample_Label sample_label : test) {
            // Compute likelihoods for the current test sample
            HashMap<String, Double> sampleLikelihoods = getLikelihood(sample_label.sample(), mapClassToSamples);

            // Compute posterior probabilities based on likelihoods and priors
            HashMap<String, Double> mapClassToPosterior = new HashMap<>();
            for (String currentClass : sampleLikelihoods.keySet()) {
                double currentClassLikelihood = sampleLikelihoods.get(currentClass);
                double posterior = currentClassLikelihood * mapClassToPriors.get(currentClass);
                mapClassToPosterior.put(currentClass, posterior);
            }

            // Sort posterior probabilities in descending order
            ArrayList<Map.Entry<String, Double>> sortPosteriors = new ArrayList<>(mapClassToPosterior.entrySet());
            sortPosteriors.sort(Map.Entry.comparingByValue(Comparator.reverseOrder()));

            // Check if the predicted class matches the true label
            int computedClass = Integer.parseInt(sortPosteriors.get(0).getKey());
            int groundTruth = Integer.parseInt(sample_label.label());
            if (computedClass == groundTruth) {
                acc++; // Increment accuracy counter
            }
        }
        acc = acc / test.size(); // Return accuracy
    }

    /**
     * This method is used to retrieve the accuracy of the testing part of the data
     * @return The accuracy
     */
    public double getAcc() {
        return acc;
    }

    /**
     * This method is used to calculate the likelihood (Density (P(X|Ci))) of the given testing sample given all classes
     * @param sample The testing sample
     * @param mapClassToSamples A map mapped each class to its samples
     * @return The map where keys are the classes and the corresponding value is the likelihood
     */
    private HashMap<String, Double> getLikelihood(double[] sample, HashMap<String, double[][]> mapClassToSamples) {
        HashMap<String, Double> mapLikelihoodPerEachClass = new HashMap<>();

        // Iterate over classes
        for (String classId : mapClassToSamples.keySet()) {
            double sum = 0.0; // Initialize sum for likelihood

            // Iterate over samples in the current class
            for (double[] samples : mapClassToSamples.get(classId)) {
                // Compute distance (scaled by bandwidth) between test sample and training sample
                for (int i = 0; i < samples.length; i++) {
                    double distanceSquared = Math.sqrt(Math.pow(sample[i] - samples[i], 2)) / bandWidth;
                    sum += gaussianKernelFunction(distanceSquared); // Compute kernel function
                }
            }

            // Compute average likelihood for the current class
            int numOfSamplesInTheCurrentClass = mapClassToSamples.get(classId).length;
            sum *= (float) 1 / numOfSamplesInTheCurrentClass;
            mapLikelihoodPerEachClass.put(classId, sum); // Store likelihood in map
        }
        return mapLikelihoodPerEachClass; // Return map of likelihoods
    }

    /**
     * This method is used to compute the Gaussian kernel function
     * @param u The squared distance among the testing sample and the samples in the training part
     * @return The results of the Gaussian kernel function
     */
    private double gaussianKernelFunction(double u) {
        return 1 / Math.sqrt(2 * Math.PI) * Math.pow(Math.E, - Math.pow(u, 2) / 2);
    }
}