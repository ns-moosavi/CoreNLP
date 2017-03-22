package edu.stanford.nlp.coref.neural;

import java.util.List;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.neural.Embedding;
import edu.stanford.nlp.neural.NeuralUtils;
import org.ejml.simple.SimpleMatrix;

public class ModelSerializer {
	  private static final String DATA_PATH =
		      "/data/nlp/moosavne/git/CoreNLP/exported_weights/";

		  public static void main(String[] args) throws Exception {
		    Embedding staticWordEmbeddings = new Embedding(DATA_PATH  + "/vectors_pretrained_all");
		    Embedding tunedWordEmbeddings = new Embedding(DATA_PATH + "/vectors_learned");

		    List<SimpleMatrix> anaphoricityModel = NeuralUtils.loadTextMatrices(
		        DATA_PATH + "/anaphoricity_weights");
		    SimpleMatrix anaBias = anaphoricityModel.remove(anaphoricityModel.size() - 1);
		    SimpleMatrix anaScale = anaphoricityModel.remove(anaphoricityModel.size() - 1);
		    anaphoricityModel.add(anaScale.mult(new SimpleMatrix(new double[][] {{-0.3}})));
		    anaphoricityModel.add(anaBias.mult(new SimpleMatrix(new double[][] {{-0.3}}))
		        .plus(new SimpleMatrix(new double[][] {{-1}})));

		    List<SimpleMatrix> pairwiseModel = NeuralUtils.loadTextMatrices(
		        DATA_PATH  + "/pairwise_weights");
		    SimpleMatrix antecedentMatrix = pairwiseModel.remove(0);
		    SimpleMatrix anaphorMatrix = pairwiseModel.remove(0);
		    SimpleMatrix pairFeaturesMatrix = pairwiseModel.remove(0);
		    SimpleMatrix pairwiseFirstLayerBias = pairwiseModel.remove(0);

		    NeuralCorefModel ncf = new NeuralCorefModel(antecedentMatrix, anaphorMatrix, pairFeaturesMatrix,
		        pairwiseFirstLayerBias, anaphoricityModel, pairwiseModel, tunedWordEmbeddings);
		    IOUtils.writeObjectToFile(ncf, DATA_PATH + "model.ser.gz");
		    IOUtils.writeObjectToFile(staticWordEmbeddings, DATA_PATH + "embeddings.ser.gz");
		  }

}
