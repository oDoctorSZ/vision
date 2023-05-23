package br.com.ia;

import br.com.ia.methods.TrainingManager;
import br.com.ia.utils.DataSplitter;
import br.com.ia.utils.Utils;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.factory.Nd4j;
import yahoofinance.Stock;
import yahoofinance.histquotes.HistoricalQuote;
import yahoofinance.histquotes.Interval;

import javax.swing.*;
import javax.xml.crypto.Data;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

public class MultiAILoader {

    public static INDArray predicted;
    public static INDArray combinedPrediction;
    public static double predictedPrice;
    public static int modelsSize = 0;
    public static int epochs = 250;
    public static int epochs2 = 1000;
    private static INDArray val;
    private static ArrayList<DayTrader> modelsFinal = new ArrayList<>();
    public static Stack<INDArray> outputStack = new Stack<>();
    private static HashMap<DayTrader, Double> predictedPrices = new HashMap<>();
    private static INDArray[] prediction = new INDArray[]{};
    private static DataSet splitTrainingDataSet = null;

    public static void Judge(List<DayTrader> models, DataSet dataSet) throws IOException, InterruptedException {

        double[] currentPrice = new double[1];
        currentPrice[0] = models.get(0).getStock().getQuote().getPrice().doubleValue();

        models.forEach(model -> {
            try {
                val = Nd4j.create(new double[] { currentPrice[0] }, new int[] {1, 1, 1});
                predictedPrice = model.getModel().output(val).getDouble(0);
                predictedPrices.put(model, predictedPrice);

            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });

        while (modelsFinal.size() < 5) {
            generateVal(models, currentPrice[0], epochs, dataSet);
        }

        predicted = Nd4j.zeros(modelsFinal.size(), 1);

        for (int j = 0; j < modelsFinal.size(); j++) {
            DayTrader model = modelsFinal.get(j);
            predicted.putScalar(j, model.getModel().output(model.getDataSet()).getDouble(0));

            model.getModel().clearLayersStates();
        }

        combinedPrediction = predicted.mean(0);
        predictedPrice = combinedPrediction.getDouble(0);

        models.forEach(model -> {
            model.setLastPrediction(predictedPrice);
        });

        if (predictedPrice > currentPrice[0]) {
            System.out.println("Comprar em " + models.get(0).getStock() + " Valor: " + predictedPrice + "!");
        } else {
            System.out.println("Vender em " + models.get(0).getStock() + " Valor: " + predictedPrice + "!");
        }

        modelsFinal.clear();
        predictedPrice = 0;
        combinedPrediction.close();

    }
    public static void TrainMode(List<DayTrader> models, Stock stock, boolean trainEnable) throws IOException {

        double[] currentPrice = new double[1];

        int sequenceLength = 15;
        int trainPercentage = 80;

        currentPrice[0] = models.get(0).getStock().getQuote().getPrice().doubleValue();


        Calendar endDate = Calendar.getInstance();
        Calendar startDate = (Calendar) endDate.clone();
        startDate.add(Calendar.YEAR, -1);
        List<HistoricalQuote> quoteData = stock.getHistory(startDate, endDate, Interval.DAILY);

        List<Calendar> dates = new ArrayList<>();
        List<Double> closePrices = new ArrayList<>();

        for (HistoricalQuote quote : quoteData) {
            dates.add(quote.getDate());
            closePrices.add(quote.getClose().doubleValue());
        }

        INDArray data = Nd4j.create(closePrices.size(), 1);
        for (int i = 0; i < closePrices.size(); i++) {
            data.putScalar(i, 0, closePrices.get(i));
        }

        val = Nd4j.create(new double[] { currentPrice[0] }, new int[] {1, 1, 1});

        XYSeriesCollection set = new XYSeriesCollection();
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Dados de treinamento vs Previsão",
                "Tempo",
                "Preço",
                set
        );

        ChartPanel chartPanel = new ChartPanel(chart);
        createChart(chartPanel, set, chart);


        if (trainEnable) {

            models.forEach(model -> {
                int k = 50;

                try {
                    splitTrainingDataSet = DataSplitter.prepareData(stock, quoteData, sequenceLength, 0.5);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                KFoldIterator iterator = new KFoldIterator(k, splitTrainingDataSet);

                for (int j = 0; j < k; j++) {
                    //models.get(0).getModel().fit(iterator);
                    iterator.next(k);
                }
            });

            for (int i = 0; i < epochs2; i++) {
                predictedPrice = 0;
                prediction = new INDArray[]{};

                models.forEach(model -> {

                    try {

                        splitTrainingDataSet = DataSplitter.prepareData(stock, quoteData, sequenceLength, 0.5);
                        //model.getModel().fit(splitTrainingDataSet);

                        ScoreIterationListener scoreListener = new ScoreIterationListener(10);
                        model.getModel().setListeners(scoreListener);

                        INDArray features = splitTrainingDataSet.getFeatures();
                        features = features.reshape(features.size(0), 1, features.size(1)); // Redimensiona a entrada

                        val = Nd4j.create(new double[] { features.getDouble() }, new int[] {1, 1, 1});
                        predictedPrice = model.getModel().output(val, true).getDouble();
                        predictedPrices.put(model, predictedPrice);

                        prediction = new INDArray[]{model.getModel().output(features)};
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
                            model.getModel().updateRnnStateWithTBPTTState();


                        }

                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }

                    //model.getModel().updateRnnStateWithTBPTTState();
                    Utils.dataSetRef.put(model.getStock(), splitTrainingDataSet);

                    try {
                        Thread.sleep(5);
                    } catch (InterruptedException e) {
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

                System.out.println(i);

            }
        }

    }


    private static void generateVal(List<DayTrader> models, double currentPrice, int times, DataSet dataSet) {
        for (int i = 0; i < times; i++) {

            predictedPrice = 0;

            models.forEach(model -> {
                predictedPrice = predictedPrices.get(model);

                if (predictedPrice > (currentPrice + 0.004) || predictedPrice < (currentPrice - 0.004)) {

                    INDArray features = dataSet.getFeatures();

                    val = Nd4j.create(new double[] { features.getDouble() }, new int[] {1, 1, 1});
                    predictedPrice = model.getModel().output(val).getDouble(0);
                    predictedPrices.replace(model, predictedPrice);

                    model.getModel().fit(dataSet);
                    model.getModel().updateRnnStateWithTBPTTState();

                    System.out.println(predictedPrice);

                } else {
                    modelsFinal.add(model);

                }

                modelsSize++;

            });

        }
    }

    public static void createChart(ChartPanel chartPanel, XYSeriesCollection set,  JFreeChart chart) {
        XYSeries trainingDataSeries = new XYSeries("Dados de Treino");
        XYSeries predictionDataSeries = new XYSeries("Previsão");
        set.addSeries(trainingDataSeries);
        set.addSeries(predictionDataSeries);

        chartPanel = new ChartPanel(chart);

        JFrame frame = new JFrame("Gráfico de Previsão");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setContentPane(chartPanel);
        frame.pack();
        frame.setVisible(true);
    }




}
