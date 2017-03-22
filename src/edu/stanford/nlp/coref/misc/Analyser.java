package edu.stanford.nlp.coref.misc;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import javax.json.Json;
import javax.json.JsonArrayBuilder;

import edu.stanford.nlp.coref.CorefDocumentProcessor;
import edu.stanford.nlp.coref.CorefProperties;
import edu.stanford.nlp.coref.CorefUtils;
import edu.stanford.nlp.coref.CorefProperties.Dataset;
import edu.stanford.nlp.coref.data.CorefCluster;
import edu.stanford.nlp.coref.data.Dictionaries;
import edu.stanford.nlp.coref.data.Document;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;

public class Analyser implements CorefDocumentProcessor {
	PrintWriter pw;
	double[][][] coref_counts;
	List<List<String>> pairs;

	public Analyser(List<List<String>> pairs) {
		coref_counts = new double[2][3][pairs.get(0).size()];
		this.pairs = pairs;
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

		Map<Pair<Integer, Integer>, Boolean> mentionPairs = CorefUtils.getLabeledMentionPairs(document);

		for (Map.Entry<Pair<Integer, Integer>, Boolean> e : mentionPairs.entrySet()) {
			Mention m1 = document.predictedMentionsByID.get(e.getKey().first);
			Mention m2 = document.predictedMentionsByID.get(e.getKey().second);
			String[] forms = new String[6];
			forms[0] = m1.spanToString() + " " + m2.spanToString();
			forms[1] = m1.headString + " " + m2.headString;
			forms[2] = getContext(m1) + " " + getContext(m2);
                        forms[3] = m2.spanToString() + " " + m1.spanToString();
                        forms[4] = m2.headString + " " + m1.headString;
                        forms[5] = getContext(m2) + " " + getContext(m1);

			int coref_index = e.getValue() ? 0 : 1;
			for (int i = 0; i < pairs.get(0).size(); i++){
				for (int j = 0; j < 3; j++)
					if (pairs.get(j).get(i).equalsIgnoreCase(forms[j]) || pairs.get(j).get(i).equalsIgnoreCase(forms[j+3]))
						coref_counts[coref_index][j][i]+=1;
			}
		}			
	}

	private String getContext (Mention m1){
		String m1Context = 
				//(m1.startIndex-2 >=0 ? m1.sentenceWords.get(m1.startIndex-2).get(CoreAnnotations.TextAnnotation.class).toLowerCase(): "NONE" )
				(m1.startIndex-1 >=0 ? m1.sentenceWords.get(m1.startIndex-1).get(CoreAnnotations.TextAnnotation.class).toLowerCase(): "NONE" )
				+ " MENTION " + 
				(m1.endIndex < m1.sentenceWords.size() ? m1.sentenceWords.get(m1.endIndex).get(CoreAnnotations.TextAnnotation.class).toLowerCase(): "NONE"); 
		//+ " " + (m1.endIndex+1 < m1.sentenceWords.size() ? m1.sentenceWords.get(m1.endIndex+1).get(CoreAnnotations.TextAnnotation.class).toLowerCase(): "NONE");

		return m1Context;
	}
	@Override
	public void finish() throws Exception {

	}



	public static void exportData(Dataset dataset, Properties props,
			Dictionaries dictionaries, List<List<String>> pairs) throws Exception {
		Analyser dataExporter = new Analyser(pairs);
		CorefProperties.setInput(props, dataset);
		dataExporter.run(props, dictionaries);
		int[] seen_num = new int[3];
		int[] only_coref_seen_num = new int[3];
		int[] only_non_coref_seen_num = new int[3];
		int[] coref_higher_seen = new int[3];
		int[] coref_non_zero_seen = new int[3];

		for (int i = 0; i < dataExporter.pairs.get(0).size(); i++){
			for (int j = 0; j < 3; j++){
				double coref_count = dataExporter.coref_counts[0][j][i];
				double non_coref_count = dataExporter.coref_counts[1][j][i];
				double prob = (coref_count+non_coref_count) > 0 ? coref_count / (coref_count+non_coref_count) : 0;
				if (coref_count != 0 || non_coref_count != 0)
					seen_num[j]++;
				if(coref_count > 0 && non_coref_count == 0)
					only_coref_seen_num[j]++;
				if(non_coref_count > 0 && coref_count == 0)
					only_non_coref_seen_num[j]++;
				if (prob > 0)
					coref_non_zero_seen[j]++;
				if(prob > 0.5)
					coref_higher_seen[j]++;
			}
		}

		int decision_num = pairs.get(0).size();
		String[] iden = {"Surface", "Head", "Context"};
		for (int i = 0; i < 3; i++){
			System.out.println(iden[i] + ": ");
			System.out.println(seen_num[i] + " out of " + decision_num + " were seen in training data");
			System.out.println("from which " + only_coref_seen_num[i] + " are only seen as coreference and " + only_non_coref_seen_num[i] + " are only seen as non-coreferent and " + coref_higher_seen[i] + " were coreferent with higher probanility and " + coref_non_zero_seen[i] + " non-zero coref probability");
			System.out.println("=========================");
		}

	}
	
	public static void extractIt(Dataset dataset, Properties props,
			Dictionaries dictionaries, List<List<String>> pairs) throws Exception {
		Analyser dataExporter = new Analyser(pairs);
		CorefProperties.setInput(props, dataset);
		dataExporter.run(props, dictionaries);
		int[] seen_num = new int[3];
		int[] only_coref_seen_num = new int[3];
		int[] only_non_coref_seen_num = new int[3];
		int[] coref_higher_seen = new int[3];
		int[] coref_non_zero_seen = new int[3];

		for (int i = 0; i < dataExporter.pairs.get(0).size(); i++){
			for (int j = 0; j < 3; j++){
				double coref_count = dataExporter.coref_counts[0][j][i];
				double non_coref_count = dataExporter.coref_counts[1][j][i];
				double prob = (coref_count+non_coref_count) > 0 ? coref_count / (coref_count+non_coref_count) : 0;
				if (coref_count != 0 || non_coref_count != 0)
					seen_num[j]++;
				if(coref_count > 0 && non_coref_count == 0)
					only_coref_seen_num[j]++;
				if(non_coref_count > 0 && coref_count == 0)
					only_non_coref_seen_num[j]++;
				if (prob > 0)
					coref_non_zero_seen[j]++;
				if(prob > 0.5)
					coref_higher_seen[j]++;
			}
		}

		int decision_num = pairs.get(0).size();
		String[] iden = {"Surface", "Head", "Context"};
		for (int i = 0; i < 3; i++){
			System.out.println(iden[i] + ": ");
			System.out.println(seen_num[i] + " out of " + decision_num + " were seen in training data");
			System.out.println("from which " + only_coref_seen_num[i] + " are only seen as coreference and " + only_non_coref_seen_num[i] + " are only seen as non-coreferent and " + coref_higher_seen[i] + " were coreferent with higher probanility and " + coref_non_zero_seen[i] + " non-zero coref probability");
			System.out.println("=========================");
		}

	}

	public static void main(String[] args) throws Exception {
		Properties props = StringUtils.argsToProperties(new String[] {"-props", args[0]});
		Dictionaries dictionaries = new Dictionaries(props);
		List<String> surfacePair = new ArrayList<String>();
		List<String> headPair = new ArrayList<String>();
		List<String> contextPair = new ArrayList<String>();
   		List<Boolean> labels = new ArrayList<Boolean>();
		List<List<String>> allPairs = new ArrayList<List<String>>();
		BufferedReader reader = new BufferedReader(new FileReader("/data/nlp/moosavne/git/CoreNLP/deep_coref_made_decisions_on_dev.txt"));
                //BufferedReader reader = new BufferedReader(new FileReader("/data/nlp/moosavne/git/current/cort/bin/cort_dev_made_decisions"));

		String line="";

		while((line = reader.readLine()) != null){
			String[] pairs = line.split("\t");
 			
			surfacePair.add(pairs[1]);
			headPair.add(pairs[2]);
			contextPair.add(pairs[3]);
		}
		allPairs.add(surfacePair);
		allPairs.add(headPair);
		allPairs.add(contextPair);

		reader.close();
		exportData(Dataset.TRAIN, props, dictionaries, allPairs);

	}
}