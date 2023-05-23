package org.silver;

import au.com.bytecode.opencsv.CSVParser;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.documentiterator.FileDocumentIterator;
import org.deeplearning4j.text.documentiterator.interoperability.DocumentIteratorConverter;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.silver.handler.TrainingHandler;
import org.silver.lstm.SilverBuilder;
import org.silver.traindata.StockData;
import yahoofinance.Stock;
import yahoofinance.YahooFinance;
import yahoofinance.histquotes.HistoricalQuote;
import yahoofinance.histquotes.Interval;

import javax.sound.midi.Sequence;
import javax.xml.crypto.Data;
import java.io.File;
import java.io.IOException;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException, InterruptedException {

        SilverBuilder model = new SilverBuilder();
        loadDataFiles(model);


    }

    public static void loadDataFiles(SilverBuilder model) throws IOException, InterruptedException {
        File trainingDataFile = new File("C:\\Users\\Ian\\SilverData\\trainingDataFile.csv");
        File testingDataFile = new File("C:\\Users\\Ian\\SilverData\\testingDataFile.csv");

        // Configurar a iteração do conjunto de dados
        int labelIndex = 1; // índice do rótulo na coluna CSV
        int batchSize = 32; // tamanho do lote para treinamento e teste
        char delimiter = ',';
        char quote = '"';
        RecordReader trainingRecordReader = new CSVRecordReader(1, delimiter, quote);
        trainingRecordReader.initialize(new FileSplit(trainingDataFile));
        RecordReaderDataSetIterator trainingIterator = new RecordReaderDataSetIterator(trainingRecordReader, batchSize, labelIndex, labelIndex + 1);
        DataNormalization scaler = new NormalizerStandardize();
        scaler.fit(trainingIterator); // ajustar normalizador com dados de treinamento
        trainingIterator.setPreProcessor(scaler);
        RecordReader testingRecordReader = new CSVRecordReader(1, delimiter, quote);
        testingRecordReader.initialize(new FileSplit(testingDataFile));
        RecordReaderDataSetIterator testingIterator = new RecordReaderDataSetIterator(testingRecordReader, batchSize, labelIndex, labelIndex + 1);
        testingIterator.setPreProcessor(scaler);

        // Configurar a iteração de documentos
        DocumentIterator trainingDocumentIterator = new FileDocumentIterator(trainingDataFile);
        DocumentIterator testingDocumentIterator = new FileDocumentIterator(testingDataFile);

        // Configurar Word2Vec
        Word2Vec word2Vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(5)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .tokenizerFactory(new DefaultTokenizerFactory())
                .iterate(trainingDocumentIterator)
                .build();
        word2Vec.fit();
        InMemoryLookupTable<VocabWord> lookupTable = (InMemoryLookupTable<VocabWord>) word2Vec.getLookupTable();
        model.getModel().setParam("W", lookupTable.getSyn0());
        model.getModel().getLayer("0").setParam("W", lookupTable.getSyn0());
        model.getModel().getLayer("0").setParam("b", Nd4j.zeros(lookupTable.getSyn0().columns()));
        model.getModel().setInput((INDArray) InputType.recurrent(lookupTable.getSyn0().columns()));

        // Treinar modelo
        for (int i = 0; i < 1000; i++) {
            while (trainingIterator.hasNext()) {
                DataSet trainingData = trainingIterator.next();
                model.getModel().fit(trainingData);
            }
            trainingIterator.reset();
        }

        // Avaliar modelo
        Evaluation evaluation = new Evaluation();
        while (testingIterator.hasNext()) {
            DataSet testingData = testingIterator.next();
            INDArray output = model.getModel().output(testingData.getFeatures());
            evaluation.eval(testingData.getLabels(), output);
        }
        System.out.println(evaluation.stats());

        // Realizar predição
        String inputText = "O sol é";
        INDArray inputVector = word2Vec.getWordVectorMatrix(inputText);
        INDArray output = model.getModel().output(inputVector);
        int predictedClass = Nd4j.argMax(output, 1).getInt(0);
        System.out.println("Texto de entrada: " + inputText);
        System.out.println("Classe prevista: " + predictedClass);
    }


    /**
     *     public void runSilver() {
     *         SilverBuilder model = new SilverBuilder();
     *         model.setStock(new Stock("AUDCAD=X"));
     *
     *         try {
     *             Calendar endDate = Calendar.getInstance();
     *             Calendar startDate = Calendar.getInstance();
     *             startDate.add(Calendar.DATE, -180);
     *             List<HistoricalQuote> historicalQuotes = model.getStock().getHistory(startDate, endDate, Interval.DAILY);
     *
     *             DataSet dataSetPreparedWithoutEqualizer = StockData.prepareData(historicalQuotes, 40, 15);
     *             DataSetIterator iterator = new ListDataSetIterator<>(dataSetPreparedWithoutEqualizer.asList(), 40);
     *             model.getModel().fit(iterator, 1000000);
     *
     *             DataSet stockDataSet = StockData.getStockDataSet(model.getStock());
     *             TrainingHandler.kFoldValidation(model, stockDataSet.getFeatures(), stockDataSet.getLabels(), 20, 100000);
     *             TrainingHandler.predictPrice(model, stockDataSet, model.getStock());
     *
     *         } catch (IOException | InterruptedException x) {
     *             x.printStackTrace();
     *         }
     *     }
     */

}


