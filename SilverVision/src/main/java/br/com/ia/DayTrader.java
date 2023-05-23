package br.com.ia;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import yahoofinance.Stock;
import yahoofinance.histquotes.HistoricalQuote;
import yahoofinance.histquotes.Interval;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class DayTrader {

    private static final int INPUT_SIZE = 1;
    private static final int OUTPUT_SIZE = 1;
    private static final int HIDDEN_SIZE = 256;
    private static final int DENSE_SIZE = 128;
    private static final int SEC_DENSE_SIZE = 64;
    private static final int THIRD_DENSE_SIZE = 32;
    private static final int FOUR_DENSE_SIZE = 16;

    public static double LEARNING_RATE = 0.15;
    private static final double L2_REGULARIZATION = 0.001;
    private static final double L1_REGULARIZATION = 0.01;
    private double lastPrediction = 0;
    private DataSet lastDataSet;

    public int timePeriod = 180;
    private MultiLayerNetwork model;
    private Stock stock;
    private MultiLayerConfiguration config;
    private static INDArray dataSet;

    public DayTrader() throws IOException {
        config = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaGrad(LEARNING_RATE))
                .l2(L2_REGULARIZATION)
                .list()
                .layer(0, new LSTM.Builder().nIn(INPUT_SIZE).nOut(HIDDEN_SIZE)
                        .activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nIn(HIDDEN_SIZE).nOut(DENSE_SIZE)
                        .activation(Activation.LEAKYRELU).build())
                .layer(2, new RnnOutputLayer.Builder().nIn(DENSE_SIZE).nOut(OUTPUT_SIZE)
                        .activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();
        model = new MultiLayerNetwork(config);
        model.init();
    }



    private double[][] getOutputLabelsMIN(Stock stock, int timePeriod) throws IOException {
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
    public void setDataSet(Stock stock) throws IOException {
        double[] output = getOutputLabels(stock);

        int numOutputs = output.length;

        double[] outputFlat = new double[numOutputs];
        for (int i = 0; i < output.length; i++) {
            outputFlat[i] = output[i];
        }

        dataSet = Nd4j.create(outputFlat, output.length);
    }

    public INDArray getDataSet() {
        return dataSet;
    }

    public Stock getStock() {
        return stock;
    }

    public void setStock(Stock stock) {
        this.stock = stock;
    }

    public MultiLayerNetwork getModel() {
        return model;
    }

    public void setModel(MultiLayerNetwork model) {
        this.model = model;
    }

    public MultiLayerConfiguration getConfig() {
        return config;
    }

    public void setConfig(MultiLayerConfiguration config) {
        this.config = config;
    }

    public boolean hasPrediction() {
        boolean debounce;
        if (lastPrediction == 0) {
            debounce = false;
        } else {
            debounce = true;
        }

        return debounce;
    }

    public double getLastPrediction() {
        return lastPrediction;
    }
    public void setLastPrediction(double lastPrediction) {
        this.lastPrediction = lastPrediction;
    }

    public DataSet getLastDataSet() {
        return lastDataSet;
    }

    public void setLastDataSet(DataSet lastDataSet) {
        this.lastDataSet = lastDataSet;
    }
}
