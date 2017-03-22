package edu.stanford.nlp.coref.misc;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
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

public class SamePairCounter implements CorefDocumentProcessor {

	double[][][] coref_counts;
	List<List<String>> pairs;
	PrintWriter pw;

	public SamePairCounter(List<List<String>> pairs, PrintWriter pw) {
		coref_counts = new double[2][3][pairs.get(0).size()];
		this.pairs = pairs;
		this.pw = pw;
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

			if (pw!= null){
				boolean are_coref = false;
				if(document.goldMentionsByID.containsKey(m1.mentionID) && document.goldMentionsByID.containsKey(m2.mentionID) &&
						document.goldMentionsByID.get(m1.mentionID).goldCorefClusterID == document.goldMentionsByID.get(m2.mentionID).goldCorefClusterID)
					are_coref = true;
				if (are_coref)
					pw.println((are_coref ? 1 : 0) + "\t"+m1.mentionType + "\t" + m2.mentionType+"\t"+m1.spanToString() + " " + m2.spanToString() + "\t" + m1.headString + " " + m2.headString + "\t"+ getContext(m1) + " " + getContext(m2));

			}
			else {
				String[] forms = new String[6];
				forms[0] = m1.spanToString() + " " + m2.spanToString();
				forms[1] = m1.headString + " " + m2.headString;
				//			forms[2] = getContext(m1) + " " + getContext(m2);
				forms[3] = m2.spanToString() + " " + m1.spanToString();
				forms[4] = m2.headString + " " + m1.headString;
				//                        forms[5] = getContext(m2) + " " + getContext(m1);

				int coref_index = e.getValue() ? 1 : 0;
				for (int i = 0; i < pairs.get(0).size(); i++){
					for (int j = 0; j < 2; j++)
						if (pairs.get(j).get(i).equalsIgnoreCase(forms[j]) || pairs.get(j).get(i).equalsIgnoreCase(forms[j+3]))
							coref_counts[coref_index][j][i]+=1;
				}
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
		PrintWriter pw = null;//new PrintWriter("all_dev_pairs.txt");
		String[] coref = {"Non-Coreference", "Coreference"};


		SamePairCounter dataExporter = new SamePairCounter(pairs, pw);
		CorefProperties.setInput(props, dataset);
		dataExporter.run(props, dictionaries);
		if (pw != null)
			pw.close();
		else {
			int[][][][] seen_num = new int[2][4][4][2];
			int[][][][] only_coref_seen_num = new int[2][4][4][2];
			int[][][][] only_non_coref_seen_num = new int[2][4][4][2];
			int[][][][] coref_higher_seen = new int[2][4][4][2];
			int[][][][] coref_non_zero_seen = new int[2][4][4][2];
			int[][][][] unseen_decisions = new int[2][4][4][2]; 
			List<String> types = Arrays.asList("PROPER", "NOMINAL", "PRONOMINAL", "LIST");
			for (int sampleNum = 0; sampleNum < dataExporter.pairs.get(0).size(); sampleNum++){
				int coref_index = Integer.parseInt(dataExporter.pairs.get(2).get(sampleNum));
				int ant_type_index = types.indexOf(dataExporter.pairs.get(3).get(sampleNum));
				int ana_type_index = types.indexOf(dataExporter.pairs.get(4).get(sampleNum));

				for (int formType = 0; formType < 2; formType++){
					double coref_count = dataExporter.coref_counts[1][formType][sampleNum];
					double non_coref_count = dataExporter.coref_counts[0][formType][sampleNum];
					double prob = (coref_count+non_coref_count) > 0 ? coref_count / (coref_count+non_coref_count) : 0;
					if (coref_count != 0 || non_coref_count != 0){
						seen_num[coref_index][ant_type_index][ana_type_index][formType]++;
						if (formType == 1)
							System.out.println("*Seen "+coref[coref_index] + " " + 
									types.get(ant_type_index)+"-"+types.get(ana_type_index)+ " " + dataExporter.pairs.get(0).get(sampleNum) + " " + coref_count + " " + non_coref_count);
					}
					if(coref_count > 0 && non_coref_count == 0)
						only_coref_seen_num[coref_index][ant_type_index][ana_type_index][formType]++;
					if(coref_count == 0 && non_coref_count == 0){
						unseen_decisions[coref_index][ant_type_index][ana_type_index][formType]++;
						//if (!dataExporter.pairs.get(1).get(sampleNum).split("\\s+")[0].equalsIgnoreCase(dataExporter.pairs.get(1).get(sampleNum).split("\\s+")[1]))
						if (formType == 1)
							System.out.println("Unseen " + coref[coref_index] + " " + 
									types.get(ant_type_index)+"-"+types.get(ana_type_index)+ " " + dataExporter.pairs.get(0).get(sampleNum) + " " + coref_count + " " + non_coref_count);
					}

					if(non_coref_count > 0 && coref_count == 0)
						only_non_coref_seen_num[coref_index][ant_type_index][ana_type_index][formType]++;
					if (prob > 0)
						coref_non_zero_seen[coref_index][ant_type_index][ana_type_index][formType]++;
					if(prob > 0.5)
						coref_higher_seen[coref_index][ant_type_index][ana_type_index][formType]++;
				}

			}

			int decision_num = pairs.get(0).size();
			int[][] type_based_decision_num = new int[4][4];
			for (int i = 0; i < pairs.get(0).size(); i++){
				type_based_decision_num[types.indexOf(pairs.get(3).get(i))][types.indexOf(pairs.get(4).get(i))]++;
			}
			String[] iden = {"Surface", "Head", "Context"};
			System.out.println("All decisions: " + decision_num);

			for (int j = 0; j < 4; j++){
				for (int k = 0; k < 4; k++){
					for (int p = 1; p < 2; p++){
						for (int i = 0; i < 2; i++){
							double seen_percent = seen_num[i][j][k][p]/(double)(seen_num[i][j][k][p]+unseen_decisions[i][j][k][p]);
							System.out.println(coref[i] + " " + types.get(j) + " " + types.get(k) + " " + iden[p]);
							System.out.println("All: " +(seen_num[i][j][k][p]+unseen_decisions[i][j][k][p])+ " Unseen: " + unseen_decisions[i][j][k][p]+ " seen percent: " + seen_percent);
							System.out.println(seen_num[i][j][k][p] + " out of " + type_based_decision_num[j][k] + " were seen in training data");
							System.out.println("from which " + only_coref_seen_num[i][j][k][p] + " are only seen as coreference and " + only_non_coref_seen_num[i][j][k][p] + " are only seen as non-coreferent and " + coref_higher_seen[i][j][k][p] + " were coreferent with higher probanility and " + coref_non_zero_seen[i][j][k][p] + " non-zero coref probability");
							System.out.println("=========================");

						}
						double sum = seen_num[0][j][k][p]+unseen_decisions[0][j][k][p]+seen_num[1][j][k][p]+unseen_decisions[1][j][k][p];
						double ratio =  (seen_num[0][j][k][p]+seen_num[1][j][k][p])/sum;
						System.out.println("All "+types.get(j) + " " + types.get(k) + " " + iden[p] + " " + sum +" from which " +  ratio + " is seen");
						System.out.println("=========================");

					}
				}
			}
		}
	}

	public static void main(String[] args) throws Exception {
		Properties props = StringUtils.argsToProperties(new String[] {"-props", args[0]});
		Dictionaries dictionaries = new Dictionaries(props);
		List<String> surfacePair = new ArrayList<String>();
		List<String> antType = new ArrayList<String>();
		List<String> anaType = new ArrayList<String>();
		List<String> headPair = new ArrayList<String>();
//		List<String> contextPair = new ArrayList<String>();
		List<String> labels = new ArrayList<String>();
		List<List<String>> allPairs = new ArrayList<List<String>>();
//		BufferedReader reader = new BufferedReader(new FileReader("/data/nlp/moosavne/git/CoreNLP/deep_coref_non_head_match_made_decisions_on_dev.txt"));
//						BufferedReader reader = new BufferedReader(new FileReader("/data/nlp/moosavne/git/current/cort/bin/cort_dev_all_made_decisions"));
//		BufferedReader reader = new BufferedReader(new FileReader("dcoref_made_decisions.txt"));
		BufferedReader reader = new BufferedReader(new FileReader("wikicoref_made_decisions.txt"));
		//		BufferedReader reader = new BufferedReader(new FileReader("all_dev_pairs.txt"));

		String line="";

		while((line = reader.readLine()) != null){
			String[] pairs = line.split("\t");
			labels.add(pairs[0].trim());
			antType.add(pairs[1]);
			anaType.add(pairs[2]);
			surfacePair.add(pairs[3]);
			headPair.add(pairs[4]);
//			contextPair.add(pairs[5]);
		}
		allPairs.add(surfacePair);
		allPairs.add(headPair);
		//		allPairs.add(contextPair);
		allPairs.add(labels);
		allPairs.add(antType);
		allPairs.add(anaType);

		reader.close();
		exportData(Dataset.TRAIN, props, dictionaries, allPairs);

	}

}

