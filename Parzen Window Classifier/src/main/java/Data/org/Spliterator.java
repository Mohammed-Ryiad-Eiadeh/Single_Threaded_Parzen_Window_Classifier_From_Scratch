package Data.org;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * This class is used to split the data into training and testing parts
 */
public class Spliterator {
    private final ArrayList<DataSource.sample_Label> data;
    private final float trainPortion;
    private final int seed;

    private List<DataSource.sample_Label> training;
    private List<DataSource.sample_Label> testing;

    /**
     * This constructor is used to creat an object of the Spliterator class
     * @param data The Dataset
     * @param trainPortion The portion of the training dataset
     * @param seed The seed to select the samples
     */
    public Spliterator(ArrayList<DataSource.sample_Label> data, float trainPortion, int seed) {
        this.data = data;
        this.trainPortion = trainPortion;
        this.seed = seed;
        this.training = new ArrayList<>();
        this.testing = new ArrayList<>();
        splitData();
    }

    /**
     * This method is used to split data into two parts, training and testing
     */
    private void splitData() {
        Random random = new Random(seed);
        Collections.shuffle(data, random);
        int splitIndex = (int) (data.size() * trainPortion);
        training = data.subList(0, splitIndex);
        testing = data.subList(splitIndex, data.size());
    }

    /**
     * This method is used to retrieve the training data
     * @return The training data
     */
    public ArrayList<DataSource.sample_Label> getTrainData() {
        return new ArrayList<>(training);
    }

    /**
     * This method is used to retrieve the testing dataset
     * @return The testing dataset
     */
    public ArrayList<DataSource.sample_Label> getTestData() {
        return new ArrayList<>(testing);
    }
}

