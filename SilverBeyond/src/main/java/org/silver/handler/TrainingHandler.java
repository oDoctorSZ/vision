package org.silver.handler;

import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.silver.lstm.SilverBuilder;
import yahoofinance.Stock;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class TrainingHandler {

    private static INDArray[] prediction = new INDArray[]{};
    private static double predictedPrice;
    private static INDArray predicted;
    private static INDArray val;


    public static void predictPrice(SilverBuilder model, DataSet dataSet, Stock stock) {

        double currentPrice = stock.getQuote().getOpen().doubleValue();

        try {
            val = Nd4j.create(new double[] { currentPrice }, new int[] {1, 1, 1});
            predictedPrice = model.getModel().output(val).getDouble(0);
            System.out.println("A" + val);
            System.out.println("B" + predictedPrice);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        INDArray output = model.getModel().output(dataSet.getFeatures());
        predictedPrice = output.getDouble(0);

        //generateVal(model, currentPrice, 10000, dataSet);

        if (predictedPrice > currentPrice) {
            System.out.println("Comprar em " + model.getStock() + " Valor: " + predictedPrice + "!");
        } else {
            System.out.println("Vender em " + model.getStock() + " Valor: " + predictedPrice + "!");
        }

        predictedPrice = 0;
    }

    public static void kFoldValidation(SilverBuilder model, INDArray features, INDArray labels, int k, int numEpochs) throws InterruptedException {
        int numSamples = (int) features.size(0);
        int numFolds = k;
        int foldSize = numSamples / numFolds;

        for (int i = 0; i < numFolds; i++) {

            int start = i * foldSize;
            int end = (i + 1) * foldSize;
            INDArray trainFeatures = Nd4j.concat(0, features.get(NDArrayIndex.interval(0, start)), features.get(NDArrayIndex.interval(end, numSamples)));
            INDArray trainLabels = Nd4j.concat(0, labels.get(NDArrayIndex.interval(0, start)), labels.get(NDArrayIndex.interval(end, numSamples)));
            INDArray valFeatures = features.get(NDArrayIndex.interval(start, end));
            INDArray valLabels = labels.get(NDArrayIndex.interval(start, end));

            for (int j = 0; j < numEpochs; j++) {
                model.getModel().fit(trainFeatures, trainLabels);
                Thread.sleep(10);
            }

            DataSet dataSet = new DataSet(valFeatures, valLabels);
            INDArray inputData = dataSet.getFeatures().reshape(1, 1, dataSet.getFeatures().size(0));
            INDArray outputData = dataSet.getLabels().reshape(dataSet.getLabels().size(0), 1);

            model.getModel().fit(inputData, outputData);
            INDArray output = model.getModel().output(inputData);
            System.out.println(output.getDouble(0));

        }
    }

    private static void generateVal(SilverBuilder model, double currentPrice, int times, DataSet dataSet) {
        double predictedPrice = Double.MAX_VALUE;
        for (int i = 0; i < times; i++) {
            if (predictedPrice > (currentPrice + 0.004) || predictedPrice < (currentPrice - 0.004)) {
                val = Nd4j.create(new double[]{ currentPrice }, new int[]{1, 1, 1});
                model.getModel().updateRnnStateWithTBPTTState();
                System.out.println(predictedPrice);
            }
        }
    }

}
