package Data.org;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;
import java.util.Scanner;

/**
 * This class is used to read the dataset (.csv) and load it
 */
public class DataSource {
    private final String dataPath;
    private final String labelID;

    /**
     * This constructor is used to creat an object of DataSource to load the dataset
     * @param dataPath The dataset path
     * @param labelID The id of the label column
     */
    public DataSource(String dataPath, String labelID) {
        this.dataPath = dataPath;
        this.labelID = labelID;
    }

    /**
     * This method is used to read the dataset
     * @return An array list containing all the data
     */
    public ArrayList<sample_Label> readData() {
        ArrayList<sample_Label> data = new ArrayList<>();
        try (Scanner scanner = new Scanner(new File(dataPath))) {
            int index = -1;
            if (scanner.hasNext()) {
                String[] headings = scanner.nextLine().split(",");
                for (int i = 0; i < headings.length; i++) {
                    if (Objects.equals(headings[i], labelID)) {
                        index = i;
                    }
                }
                if (index == -1) {
                    throw new IllegalArgumentException("The id of the provided class does not exist");
                }
            }
            while (scanner.hasNext()) {
                String[] row = scanner.nextLine().split(",");
                String label = row[index];
                String[] sample = new String[row.length - 1];
                int column = 0;
                for (int i = 0; i < sample.length; i++) {
                    column = i;
                    if (column == index) {
                        column++;
                    }
                    sample[i] = row[column];
                }
                double[] vector = Arrays.stream(sample).mapToDouble(Double::parseDouble).toArray();
                data.add(new sample_Label(vector, label));
            }
        } catch (IOException exception) {
            exception.printStackTrace();
        }
        return data;
    }

    /**
     * This record is used to creat an object holds an entry of sample (feature vec) and the corresponding label (class)
     */
    public record sample_Label(double[] sample, String label) { }
}
