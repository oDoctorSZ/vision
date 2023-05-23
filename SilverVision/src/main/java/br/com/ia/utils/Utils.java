package br.com.ia.utils;

import br.com.ia.DayTrader;
import br.com.ia.MultiAILoader;
import br.com.ia.SilverVision;
import br.com.ia.methods.TrainingManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import yahoofinance.Stock;
import yahoofinance.YahooFinance;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class Utils {

    public static List<DayTrader> models = new ArrayList<>();
    public static HashMap<Stock, DataSet> dataSetRef = new HashMap<>();
    private static HashMap<Stock, List<DayTrader>> modelStockHash = new HashMap<>();
    private static boolean debounce = false;
    static int SILVER_INSTANCES = 1;

    private static final List<String> actionsContainer = Arrays.asList("AUDCAD=X", "AUDCHF=X", "AUDJPY=X", "BRLUSD=X",
            "CADCHF=X", "CHFJPY=X", "EURAUD=X", "EURCAD=X", "EURCHF=X", "EURGBP=X",
            "GBPAUD=X", "GBPJPY=X", "NZDCAD=X", "NZDCHF=X", "AUDUSD=X", "EURUSD=X", "EURJPY=X");

    private static final List<String> actionsContainerReduced = Arrays.asList("AUDCAD=X", "AUDCHF=X", "AUDJPY=X");

    public static void initializeMarketStock() {
        actionsContainerReduced.forEach(stock -> {
            try {
                //TrainingManager.loadDataSet(YahooFinance.get(stock));
                System.out.println(stock + " Loaded!");
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });

        System.out.println("DataSet Loaded!");
    }

    public static void initializerSilverTrainer() throws Exception {
        for (int i = 0; i < SILVER_INSTANCES; i++) {
            DayTrader md = new DayTrader();
            //TrainingManager.trainWithDataSet(md.getModel());
            models.add(md);
        }

        debounce = true;
    }

    public static void silverJudgeWithMultiAI(String curr, boolean trainEnable) throws Exception {

        Stock stock = YahooFinance.get(curr);
        DataSet dataSet = TrainingManager.getStockDataSet(stock);
        System.out.println(dataSet);

        if (modelStockHash.containsKey(stock)) {
            models.clear();
            models = modelStockHash.get(stock);

        } else {
            //initializerSilverEachStock(stock, models);
        }

        if (debounce) {
            models.forEach(model -> {
                try {
                    model.setStock(stock);
                    //model.setDataSet(stock);
                    model.getModel().fit(dataSet);

                } catch (Exception x) {
                    x.getStackTrace();
                }
            });
        }

        MultiAILoader.TrainMode(models, stock, trainEnable);

        /*
                actionsContainerReduced.forEach(act -> {
            try {
                MultiAILoader.TrainMode(models, YahooFinance.get(act), trainEnable);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
         */


        MultiAILoader.Judge(models, dataSet);
    }


    private static void initializerSilverEachStock(Stock stock, List<DayTrader> modelList) throws IOException {

        modelList.clear();

        for (int i = 0; i < SILVER_INSTANCES; i++) {
            DayTrader md = new DayTrader();
            md.setStock(stock);
            //md.setDataSet(stock);

            modelList.add(md);
            modelStockHash.put(stock, modelList);
        }

        debounce = true;
    }


    public static void newReward(DayTrader model, double currentPrice, double predictedPrice) {
        double rew = SilverVision.reward(currentPrice, predictedPrice, DayTrader.LEARNING_RATE);

        INDArray reward = Nd4j.create(new double[] { rew });
        reward = reward.reshape(1, 1, -1); // ajusta a forma para [1,1,1] -> [1,1,1120]
        INDArray w = model.getModel().getLayer(5).getParam("W");
        INDArray newW = w.sub(reward.broadcast(w.shape()));
        model.getModel().getLayer(5).setParam("W", newW); // atribui a nova matriz ao parâmetro "W"
    }





}
