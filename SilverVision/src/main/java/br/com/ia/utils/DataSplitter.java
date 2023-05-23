package br.com.ia.utils;
import br.com.ia.DayTrader;
import br.com.ia.MultiAILoader;
import jdk.jshell.execution.Util;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import yahoofinance.Stock;
import yahoofinance.histquotes.HistoricalQuote;
import yahoofinance.histquotes.Interval;

import javax.swing.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.List;

public class DataSplitter {

    public static JFrame frame = new JFrame("Gráficos de Treinamento");

    public static DataSet prepareDataSplit(Stock stock,List<HistoricalQuote> quoteData, int sequenceLength, double splitRatio) throws IOException {

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

        int testSize = (int) Math.round((1 - splitRatio) * closePrices.size());
        int numExamples = closePrices.size() - sequenceLength + 1;
        int numTestExamples = testSize - sequenceLength + 1;
        int numTrainExamples = numExamples - numTestExamples;

        INDArray input = Nd4j.create(numExamples, sequenceLength, 1);
        INDArray output = Nd4j.create(numExamples, 1);

        for (int i = 0; i < numExamples; i++) {
            for (int j = 0; j < sequenceLength; j++) {
                input.putScalar(new int[]{i, j, 0}, data.getDouble(i + j, 0));
            }
            output.putScalar(i, 0, data.getDouble(i + sequenceLength - 1, 0));
        }

        INDArray[] inputSplit = new INDArray[]{
                input.get(NDArrayIndex.interval(0, numTrainExamples), NDArrayIndex.all(), NDArrayIndex.all()),
                input.get(NDArrayIndex.interval(numTrainExamples, numExamples), NDArrayIndex.all(), NDArrayIndex.all())
        };
        INDArray[] outputSplit = new INDArray[]{
                output.get(NDArrayIndex.interval(0, numTrainExamples), NDArrayIndex.all()),
                output.get(NDArrayIndex.interval(numTrainExamples, numExamples), NDArrayIndex.all())
        };

        return new DataSet(inputSplit[0], outputSplit[0]);
    }

    public static DataSet prepareData(Stock stock, List<HistoricalQuote> quoteData, int sequenceLength, double splitRatio) throws IOException {
        List<Calendar> dates = new ArrayList<>();
        List<Double> closePrices = new ArrayList<>();

        // Obter os preços de fechamento
        for (HistoricalQuote quote : quoteData) {
            dates.add(quote.getDate());
            closePrices.add(quote.getClose().doubleValue());
        }

        // Normalizar os dados
        double max = Collections.max(closePrices);
        double min = Collections.min(closePrices);
        for (int i = 0; i < closePrices.size(); i++) {
            closePrices.set(i, (closePrices.get(i) - min) / (max - min));
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

        return new DataSet(input, output);
    }


    public static DataSet prepareData15Min(List<HistoricalQuote> quoteData, int sequenceLength, int trainSize ,double splitRatio) {
        List<Long> dates = new ArrayList<>();
        List<Double> closePrices = new ArrayList<>();

        for (HistoricalQuote quote : quoteData) {
            dates.add(quote.getDate().getTimeInMillis());
            closePrices.add(quote.getClose().doubleValue());
        }

        INDArray data = Nd4j.create(closePrices.size(), 1);
        for (int i = 0; i < closePrices.size(); i++) {
            data.putScalar(i, 0, closePrices.get(i));
        }

        int testSize = (int) Math.round((1 - splitRatio) * closePrices.size());
        int numExamples = closePrices.size() - sequenceLength + 1;
        int numTestExamples = Math.min(numExamples - trainSize, Math.max(0, testSize - sequenceLength + 1));
        int numTrainExamples = numExamples - numTestExamples;

        INDArray input = Nd4j.create(numExamples, sequenceLength, 1);
        INDArray output = Nd4j.create(numExamples, 1);

        for (int i = 0; i < numExamples; i++) {
            if (i + sequenceLength < data.length()) {
                for (int j = 0; j < sequenceLength; j++) {
                    input.putScalar(new int[] {i, j, 0}, data.getDouble(i + j, 0));
                }
                output.putScalar(i, 0, data.getDouble(i + sequenceLength, 0));
            }
        }

        INDArray[] inputSplit = new INDArray[] {
                input.get(NDArrayIndex.interval(0, numTrainExamples), NDArrayIndex.all(), NDArrayIndex.all()),
                input.get(NDArrayIndex.interval(numTrainExamples, numExamples), NDArrayIndex.all(), NDArrayIndex.all())
        };
        INDArray[] outputSplit = new INDArray[] {
                output.get(NDArrayIndex.interval(0, numTrainExamples), NDArrayIndex.all()),
                output.get(NDArrayIndex.interval(numTrainExamples, numExamples), NDArrayIndex.all())
        };

        //NormalizerStandardize normalizer = new NormalizerStandardize();
        //DataSet dataSet = new DataSet(input, output);
        //normalizer.fit(dataSet);
        //normalizer.transform(data);

        Evaluation eval = new Evaluation();
        eval.eval(inputSplit[0].reshape(inputSplit[0].size(0), -1), inputSplit[0].reshape(inputSplit[0].size(0), -1));
        System.out.println(eval.stats());

        return new DataSet(inputSplit[0], outputSplit[0]);
    }

}
