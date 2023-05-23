package org.silver.lstm;

import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import yahoofinance.Stock;

public class SilverBuilder {

    private static final int INPUT_SIZE = 3;
    private static final int OUTPUT_SIZE = 1;
    private static final int DENSE_SIZE = 280;
    private static final int SEC_DENSE_SIZE = 150;
    private static final int THIRD_DENSE_SIZE = 80;
    private static final int FOUR_DENSE_SIZE = 64;
    private static final int FIVE_DENSE_SIZE = 150;
    private static final int SIX_DENSE_SIZE = 80;
    private static final int SEVEN_DENSE_SIZE = 64;
    private static final int EIGHT_DENSE_SIZE = 32;
    private static double LEARNING_RATE = 0.01;
    private static final double L2_REGULARIZATION = 0.001;
    private MultiLayerNetwork model;
    private MultiLayerConfiguration modelConfig;
    private Stock stock;

    public SilverBuilder() {
        modelConfig = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(LEARNING_RATE))
                .l2(L2_REGULARIZATION)
                .list()
                .layer(0, new LSTM.Builder().nIn(INPUT_SIZE).nOut(DENSE_SIZE).activation(Activation.TANH).build())
                .layer(1, new DenseLayer.Builder().nIn(DENSE_SIZE).nOut(SEC_DENSE_SIZE).activation(Activation.LEAKYRELU).build())
                .layer(2, new DenseLayer.Builder().nIn(SEC_DENSE_SIZE).nOut(THIRD_DENSE_SIZE).activation(Activation.LEAKYRELU).dropOut(0.5).build())
                .layer(3, new DenseLayer.Builder().nIn(THIRD_DENSE_SIZE).nOut(FOUR_DENSE_SIZE).activation(Activation.LEAKYRELU).build())
                .layer(4, new RnnOutputLayer.Builder().nIn(FOUR_DENSE_SIZE).nOut(OUTPUT_SIZE).lossFunction(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTLength(10)
                .build();

        model = new MultiLayerNetwork(modelConfig);
        model.init();
    }

    public MultiLayerNetwork getModel() {
        return model;
    }

    public void setModel(MultiLayerNetwork model) {
        this.model = model;
    }

    public Stock getStock() {
        return stock;
    }

    public void setStock(Stock stock) {
        this.stock = stock;
    }
}
