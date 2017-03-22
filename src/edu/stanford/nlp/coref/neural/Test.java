package edu.stanford.nlp.coref.neural;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import javax.json.Json;
import javax.json.JsonArrayBuilder;
import javax.json.JsonObject;
import javax.json.JsonObjectBuilder;

import edu.stanford.nlp.coref.CorefDocumentProcessor;
import edu.stanford.nlp.coref.CorefProperties;
import edu.stanford.nlp.coref.CorefUtils;
import edu.stanford.nlp.coref.CorefProperties.Dataset;
import edu.stanford.nlp.coref.data.CorefCluster;
import edu.stanford.nlp.coref.data.Dictionaries;
import edu.stanford.nlp.coref.data.Document;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.coref.data.Document.DocType;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.ud.CoNLLUUtils.FeatureNameComparator;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;

public class Test implements CorefDocumentProcessor {
	private final boolean conll;
	private final PrintWriter dataWriter;
	private final PrintWriter goldClusterWriter;
	private final Dictionaries dictionaries;
	private final boolean useExtendedPairwiseFeatures = true;
	private CategoricalFeatureExtractor extractor;
	double[] featurePosCounts = null;
	double[] featNegCounts = null;
	public Test(Properties props, Dictionaries dictionaries, String dataPath,
			String goldClusterPath) {
		conll = CorefProperties.conll(props);
		extractor = new CategoricalFeatureExtractor(props, dictionaries);

		this.dictionaries = dictionaries;
		try {
			dataWriter = IOUtils.getPrintWriter(dataPath);
			goldClusterWriter = IOUtils.getPrintWriter(goldClusterPath);
		} catch (Exception e) {
			throw new RuntimeException("Error creating data exporter", e);
		}
	}

	@Override
	public void process(int id, Document document) {
		JsonArrayBuilder clusters = Json.createArrayBuilder();
		for (CorefCluster gold : document.goldCorefClusters.values()) {
			JsonArrayBuilder c = Json.createArrayBuilder();
			for (Mention m : gold.corefMentions) {
				c.add(m.mentionID);
			}
			clusters.add(c.build());
		}
		goldClusterWriter.println(Json.createObjectBuilder().add(String.valueOf(id),
				clusters.build()).build());

		Map<Pair<Integer, Integer>, Boolean> mentionPairs = CorefUtils.getLabeledMentionPairs(document);
		List<Mention> mentionsList = CorefUtils.getSortedMentions(document);
		Map<Integer, List<Mention>> mentionsByHeadIndex = new HashMap<>();
		for (int i = 0; i < mentionsList.size(); i++) {
			Mention m = mentionsList.get(i);
			List<Mention> withIndex = mentionsByHeadIndex.get(m.headIndex);
			if (withIndex == null) {
				withIndex = new ArrayList<>();
				mentionsByHeadIndex.put(m.headIndex, withIndex);
			}
			withIndex.add(m);
		}

		if (useExtendedPairwiseFeatures){
			Map<Integer, Map<Integer, List<Integer>>> allFeatures = 
					extractor.extendedPairwiseFeatures(document, mentionsList, dictionaries, conll, null);
			if (featurePosCounts == null){
				featurePosCounts = new double[extractor.parser.m_proFeatures.size()+6];
				featNegCounts = new double[extractor.parser.m_proFeatures.size()+6];
			}

			for (Map.Entry<Pair<Integer, Integer>, Boolean> e : mentionPairs.entrySet()) {
				List<Integer> pairwiseFeatures = allFeatures.get(e.getKey().second).get(e.getKey().first);

				for (int i =0; i < pairwiseFeatures.size(); i++) {
					if (pairwiseFeatures.get(i) == 1){
						if (e.getValue())
							featurePosCounts[i]++;
						else
							featNegCounts[i]++;
					}
				}
			}
		}
	}

	@Override
	public void finish() throws Exception {
		double[] prob = new double[featurePosCounts.length];
		for (int i = 0; i < prob.length; i++)
			prob[i] = featurePosCounts[i]/(featurePosCounts[i]+featNegCounts[i]);
		System.out.println(Arrays.toString(featurePosCounts));
		System.out.println(Arrays.toString(featNegCounts));
		System.out.println(Arrays.toString(prob));

	}

	public static void exportData(String outputPath, Dataset dataset, Properties props,
			Dictionaries dictionaries) throws Exception {
		CorefProperties.setInput(props, dataset);
		String dataPath = outputPath + "/data_raw/";
		String goldClusterPath = outputPath + "/gold/";
		IOUtils.ensureDir(new File(dataPath));
		IOUtils.ensureDir(new File(goldClusterPath));
		new Test(props, dictionaries,
				dataPath + dataset.toString().toLowerCase(),
				goldClusterPath + dataset.toString().toLowerCase()).run(props, dictionaries);
	}

	public static void main(String[] args) throws Exception {
		double i = -0.3;
		System.out.println((i%1));
//		Properties props = StringUtils.argsToProperties(new String[] {"-props", args[0]});
//		Dictionaries dictionaries = new Dictionaries(props);
//		String outputPath = args[1];
//		exportData(outputPath, Dataset.TEST, props, dictionaries);
	}
}
