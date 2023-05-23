package br.com.ia.methods;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import yahoofinance.Stock;
import yahoofinance.histquotes.HistoricalQuote;
import yahoofinance.histquotes.Interval;

import javax.swing.*;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class TrainingManager {


    public static HashMap<String, DataSet> dataSetMap = new HashMap<>();
    public static List<DataSet> dataSets = new ArrayList<>();
    public static Stack<List<HistoricalQuote>> quoteDataStack = new Stack<>();

    public static JFrame frame = new JFrame("Gráficos de Treinamentoo");

    public static void loadDataSet(Stock stock) throws Exception {
        double[][] input = getTrainingData(stock);
        double[] output = getOutputLabels(stock);

        int numInputs = input.length * input[0].length;
        int numOutputs = output.length;

        double[] inputFlat = new double[numInputs];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[i].length; j++) {
                inputFlat[i * input[i].length + j] = input[i][j];
            }
        }

        double[] outputFlat = new double[numOutputs];
        for (int i = 0; i < output.length; i++) {
            outputFlat[i] = output[i];
        }

        INDArray inputND = Nd4j.create(inputFlat);
        INDArray outputND = Nd4j.create(outputFlat);

        DataSet dataSet = new DataSet(inputND, outputND);

        dataSetMap.put(stock.getName(), dataSet);
        dataSets.add(dataSet);
    }

    public static void trainWithDataSet(MultiLayerNetwork model) {

        dataSetMap.forEach((stockName, dataSet) -> {
            model.fit(dataSet);
        });

        System.out.println("Silver Instance has been trained with all the market actions!");

    }

    public static DataSet getStockDataSet(Stock stock) throws IOException {
        double[][] input = getTrainingData(stock);
        double[] output = getOutputLabels(stock);

        int numInputs = input.length * input[0].length;
        int numOutputs = output.length;

        double[] inputFlat = new double[numInputs];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[i].length; j++) {
                inputFlat[i * input[i].length + j] = input[i][j];
            }
        }

        double[] outputFlat = new double[numOutputs];
        System.arraycopy(output, 0, outputFlat, 0, output.length);

        INDArray inputND = Nd4j.create(inputFlat);
        INDArray outputND = Nd4j.create(outputFlat);

        return new DataSet(inputND, outputND);
    }

    public static double[][] normalizeData(double[][] data) {
        double[][] normalizedData = new double[data.length][data[0].length];
        double max = getMax(data);
        double min = getMin(data);
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                normalizedData[i][j] = (data[i][j] - min) / (max - min);
            }
        }
        return normalizedData;
    }

    public static double getMax(double[][] data) {
        double max = Double.MIN_VALUE;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                if (data[i][j] > max) {
                    max = data[i][j];
                }
            }
        }
        return max;
    }

    public static double getMin(double[][] data) {
        double min = Double.MAX_VALUE;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                if (data[i][j] < min) {
                    min = data[i][j];
                }
            }
        }
        return min;
    }

    public static double[] getOutputLabels(Stock stock) throws IOException {
        Calendar endDate = Calendar.getInstance();
        Calendar startDate = (Calendar) endDate.clone();
        startDate.add(Calendar.YEAR, -1);
        List<HistoricalQuote> quotes = stock.getHistory(startDate, endDate, Interval.DAILY);

        int numQuotes = quotes.size();
        double[] outputLabels = new double[numQuotes];
        for (int i = 0; i < numQuotes; i++) {
            outputLabels[i] = quotes.get(i).getClose().doubleValue();
        }
        return outputLabels;
    }

    public static double[][] getTrainingData(Stock stock) throws IOException {

        Calendar from = Calendar.getInstance();
        from.add(Calendar.YEAR, -1);
        Calendar to = Calendar.getInstance();
        List<HistoricalQuote> quotes = stock.getHistory(from, to, Interval.DAILY);

        int numQuotes = quotes.size();
        double[][] trainingData = new double[numQuotes][1];
        for (int i = 0; i < numQuotes; i++) {
            trainingData[i][0] = quotes.get(i).getClose().doubleValue();
        }
        return trainingData;
    }

    public static double[][] getTrainingDataMIN(Stock stock) throws IOException {

        Calendar from = Calendar.getInstance();
        from.add(Calendar.YEAR, -1);

        Calendar to = Calendar.getInstance();

        List<HistoricalQuote> quotes = stock.getHistory(from, to, Interval.DAILY);

        Map<String, List<HistoricalQuote>> quotesByDate = quotes.stream()
                .collect(Collectors.groupingBy(q -> q.getDate().toString()));

        Map<String, List<HistoricalQuote>> quotesBy15Min = new HashMap<>();

        quotesByDate.forEach((date, dailyQuotes) -> {
            List<HistoricalQuote> quotes15Min = new ArrayList<>();

            for (int i = 0; i < dailyQuotes.size(); i += 4) {
                HistoricalQuote quote = dailyQuotes.get(i);
                quotes15Min.add(quote);
            }

            quotesBy15Min.put(date, quotes15Min);
        });

        int numQuotes = quotesBy15Min.values().size();
        double[][] trainingData = new double[1][numQuotes];
        for (int i = 0; i < numQuotes; i++) {
            trainingData[0][i] = quotes.get(i).getClose().doubleValue();
        }


        return trainingData;
    }

    public static double[][] getOutputLabelsMIN(Stock stock) throws IOException {
        // Recupere os preços de fechamento das ações nos próximos "timePeriod" dias
        Calendar from = Calendar.getInstance();
        from.add(Calendar.YEAR, -1);

        Calendar to = Calendar.getInstance();

        List<HistoricalQuote> quotes = stock.getHistory(from, to, Interval.DAILY);

        Map<String, List<HistoricalQuote>> quotesByDate = quotes.stream()
                .collect(Collectors.groupingBy(q -> q.getDate().toString()));

        Map<String, List<HistoricalQuote>> quotesBy15Min = new HashMap<>();

        quotesByDate.forEach((date, dailyQuotes) -> {
            List<HistoricalQuote> quotes15Min = new ArrayList<>();

            for (int i = 0; i < dailyQuotes.size(); i += 4) {
                HistoricalQuote quote = dailyQuotes.get(i);
                quotes15Min.add(quote);
            }

            quotesBy15Min.put(date, quotes15Min);
        });

        int numQuotes = quotesBy15Min.values().size();
        double[][] outputLabels = new double[1][numQuotes];
        for (int i = 0; i < numQuotes; i++) {
            outputLabels[0][i] = quotes.get(i).getClose().doubleValue();
        }

        return outputLabels;
    }

    /*
    models.forEach(model -> {

                    // Prepare the data
                    try {
                        splitTrainingDataSet = DataSplitter.prepareDataSplit(stock, quoteData, sequenceLength, 0.5);
                        model.getModel().fit(splitTrainingDataSet);
                        splitTrainingDataSet = DataSplitter.prepareData(stock, quoteData, sequenceLength, 0);
                        model.getModel().fit(splitTrainingDataSet);

                        ScoreIterationListener scoreListener = new ScoreIterationListener(1);
                        model.getModel().setListeners(scoreListener);

                        INDArray features = splitTrainingDataSet.getFeatures();
                        features = features.reshape(features.size(0), 1, features.size(1)); // Redimensiona a entrada

                        val = Nd4j.create(new double[] { features.getDouble() }, new int[] {1, 1, 1});
                        predictedPrice = models.get(0).getModel().output(val).getDouble(0);
                        predictedPrices.put(models.get(0), predictedPrice);

                        prediction = new INDArray[]{models.get(0).getModel().output(features)};
                        INDArray yPrediction = prediction[0];

                        // Plot the training data and prediction
                        XYSeries trainingDataSeries = set.getSeries(0);
                        XYSeries predictionDataSeries = set.getSeries(1);
                        trainingDataSeries.clear();
                        predictionDataSeries.clear();

                        for (int j = 0; j < splitTrainingDataSet.getFeatures().size(0); j++) {
                            double x = splitTrainingDataSet.getFeatures().getDouble(j, 0, 0);
                            double yTraining = data.getDouble(j + sequenceLength - 1);
                            trainingDataSeries.add(x, yTraining);

                            double yPredictionValue = yPrediction.getDouble(j);
                            predictionDataSeries.add(x, yPredictionValue);

                            chartPanel.repaint();

                        }


                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }

                    model.getModel().updateRnnStateWithTBPTTState();
                    Utils.dataSetRef.put(model.getStock(), splitTrainingDataSet);

                    try {
                        Thread.sleep(5);
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                });
     */


}
