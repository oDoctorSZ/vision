package org.silver.traindata;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import yahoofinance.Stock;
import yahoofinance.histquotes.HistoricalQuote;
import yahoofinance.histquotes.Interval;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.List;

public class StockData {

    public static DataSet prepareData(List<HistoricalQuote> quoteData, int sequenceLength, double splitRatio) throws IOException {
        List<Calendar> dates = new ArrayList<>();
        List<Double> closePrices = new ArrayList<>();

        // Obter os preços de fechamento
        for (HistoricalQuote quote : quoteData) {
            dates.add(quote.getDate());
            closePrices.add(quote.getClose().doubleValue());
        }

        // Converter a lista de preços normalizados em um array INDArray
        INDArray data = Nd4j.create(closePrices.size(), 1);
        for (int i = 0; i < closePrices.size(); i++) {
            data.putScalar(i, 0, closePrices.get(i));
        }

        // Dividir os dados em conjuntos de treinamento e teste
        int testSize = (int) Math.round((1 - splitRatio) * closePrices.size());
        int numExamples = closePrices.size() - sequenceLength + 1;
        int numTestExamples = testSize - sequenceLength + 1;
        int numTrainExamples = numExamples - numTestExamples;

        // Criar os conjuntos de entrada e saída
        INDArray input = Nd4j.create(numExamples, 1, sequenceLength);
        INDArray output = Nd4j.create(numExamples, 1);

        for (int i = 0; i < numExamples; i++) {
            for (int j = 0; j < sequenceLength; j++) {
                input.putScalar(new int[]{i, j, 0}, data.getDouble(i + j, 0));
            }
            output.putScalar(i, 0, data.getDouble(i + sequenceLength - 1, 0));
        }

        DataSet vs = new DataSet(input, output);
        System.out.println(vs);
        return vs;
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

    private static double[] getOutputLabels(Stock stock) throws IOException {
        Calendar endDate = Calendar.getInstance();
        Calendar startDate = Calendar.getInstance();
        startDate.add(Calendar.DATE, -180);
        List<HistoricalQuote> quotes = stock.getHistory(startDate, endDate, Interval.DAILY);

        int numQuotes = quotes.size();
        double[] outputLabels = new double[numQuotes];
        for (int i = 0; i < numQuotes; i++) {
            outputLabels[i] = quotes.get(i).getClose().doubleValue();
        }
        return outputLabels;
    }

    private static double[][] getTrainingData(Stock stock) throws IOException {
        Calendar to = Calendar.getInstance();
        Calendar from = Calendar.getInstance();
        from.add(Calendar.DATE, -180);
        List<HistoricalQuote> quotes = stock.getHistory(from, to, Interval.DAILY);

        int numQuotes = quotes.size();
        double[][] trainingData = new double[numQuotes][1];
        for (int i = 0; i < numQuotes; i++) {
            trainingData[i][0] = quotes.get(i).getClose().doubleValue();
        }
        return trainingData;
    }

}
