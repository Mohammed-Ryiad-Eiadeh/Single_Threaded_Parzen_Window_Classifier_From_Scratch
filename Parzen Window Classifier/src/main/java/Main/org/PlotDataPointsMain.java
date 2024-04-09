package Main.org;

import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.ScatterPlot;

public class PlotDataPointsMain {
    public static void main(String[] args) {
        // the path of the generated data
        String dataURL = "...\\data.csv";

        // read the data by object of Table
        var dataTable = Table.read().csv(dataURL);

        // plot the scatter plot of the given two features according to the label
        Plot.show(ScatterPlot.create("Plot", dataTable, "F0", "F1", "class"));
    }
}
