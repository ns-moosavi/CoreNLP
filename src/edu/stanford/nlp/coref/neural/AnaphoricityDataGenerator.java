package edu.stanford.nlp.coref.neural;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Pattern;

import edu.stanford.nlp.coref.CorefDocumentProcessor;
import edu.stanford.nlp.coref.CorefProperties;
import edu.stanford.nlp.coref.CorefProperties.Dataset;
import edu.stanford.nlp.coref.CorefUtils;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.coref.data.CorefCluster;
import edu.stanford.nlp.coref.data.Dictionaries;
import edu.stanford.nlp.coref.data.Document;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.coref.data.Dictionaries.MentionType;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;

/**
 * Outputs the CoNLL CoNLL data for training a neural coreference system
 * (implented in python/theano).
 * See https://github.com/clarkkev/deep-coref for the training code.
 * @author Kevin Clark
 */
public class AnaphoricityDataGenerator implements CorefDocumentProcessor {
	private final boolean conll;
	private final PrintWriter dataWriter;
	private final Dictionaries dictionaries;
	private final Map<String, Map<Integer, Double>> anaScores;


	public AnaphoricityDataGenerator(Properties props, Dictionaries dictionaries, String dataPath, Map<String, Map<Integer, Double>> anaphoricityScores) {
		conll = CorefProperties.conll(props);
		this.dictionaries = dictionaries;
		this.anaScores = anaphoricityScores;

		try {
			dataWriter = IOUtils.getPrintWriter(dataPath);
		} catch (Exception e) {
			throw new RuntimeException("Error creating data exporter", e);
		}
	}

	@Override
	public void process(int id, Document document) {

		List<Set<Integer>> existentialFeatures = new ArrayList<Set<Integer>>();
		List<Set<Integer>> prevContextExistentialFeatures = new ArrayList<Set<Integer>>();
		document.extractGoldCorefClusters();

		int featureNum = 6;
		for (int i = 0; i < featureNum; i++){
			existentialFeatures.add(new HashSet<Integer>());
			prevContextExistentialFeatures.add(new HashSet<Integer>());
		}
		List<Mention> mentionsList = CorefUtils.getSortedMentions(document);

		Set<Integer> allFirstMentions = new HashSet<Integer>();
		for (CorefCluster c : document.goldCorefClusters.values()){
			List<Mention> sortedMentions = new ArrayList<>(c.corefMentions.size());
			sortedMentions.addAll(c.corefMentions);
			Collections.sort(sortedMentions, (m1, m2) -> m1.appearEarlierThan(m2) ? -1 : 1);

			allFirstMentions.add(sortedMentions.get(0).mentionID);
		}

		Set<Integer> allCoreferentMentions = new HashSet<Integer>();
        for (List<Mention> ml : document.goldMentions){
        	for (Mention m : ml)
        		allCoreferentMentions.add(m.mentionID);
        	
        }
        
//		Map<Pair<Integer, Integer>, Boolean> mentionPairs = CorefUtils.getLabeledMentionPairs(document);
//		Map<Integer, List<Mention>> mentionsByHeadIndex = new HashMap<>();
//		for (int i = 0; i < mentionsList.size(); i++) {
//			Mention m = mentionsList.get(i);
//			List<Mention> withIndex = mentionsByHeadIndex.get(m.headIndex);
//			if (withIndex == null) {
//				withIndex = new ArrayList<>();
//				mentionsByHeadIndex.put(m.headIndex, withIndex);
//			}
//			withIndex.add(m);
//		}
//
//		for (Map.Entry<Pair<Integer, Integer>, Boolean> e : mentionPairs.entrySet()) {
//			Mention anaphor = document.predictedMentionsByID.get(e.getKey().first);
//			Mention ante = document.predictedMentionsByID.get(e.getKey().second);
//
//			if (e.getValue()){
//				allCoreferentMentions.add(anaphor.mentionID);
//				allCoreferentMentions.add(ante.mentionID);
//			}
//		}


		String separator = " QQQQQQQQQQ ";

		for (Mention m : mentionsList) {
		      Iterator<SemanticGraphEdge> iterator =
		              m.enhancedDependency.incomingEdgeIterator(m.headIndexedWord);
		          SemanticGraphEdge relation = iterator.hasNext() ? iterator.next() : null;
		          String depParent = relation == null ? "<missing>" : relation.getSource().word();

			boolean[] featVals =  CategoricalFeatureExtractor.anaphoricityFeatures(
					document, m, mentionsList, dictionaries);

			String line = document.conllDoc.documentID +document.conllDoc.getPartNo()+ separator + m.mentionID + separator + m.contextRepresentation(document)+separator;
			line += m.spanToString().toLowerCase() + separator;
			line += m.headString.toLowerCase() ;

			for (int i = 0; i < featVals.length; i++){
				if (featVals[i])
					line+= separator + "1";
				else
					line+= separator + "0";
			}

//			line += separator + (mentionsByHeadIndex.get(m.headIndex).stream()
//					.anyMatch(m2 -> m != m2 && m.insideIn(m2)) ? "1" : "0");

			line += separator + m.mentionType;
			line  += separator + m.mentionNum/(float)(mentionsList.size());
			line  += separator + (anaScores.get(document.conllDoc.documentID+ document.conllDoc.getPartNo()).get(m.mentionID)+1)/2.0;
			line += separator + fine_type(m);
			line += separator + depParent;
			line += separator + (relation == null ? "<missing>" : relation.getRelation().getShortName());

			int label = 0;
// 
			if ( allCoreferentMentions.contains(m.mentionID) && !allFirstMentions.contains(m.mentionID))
				label = 1;
			else if (allCoreferentMentions.contains(m.mentionID))
				label = 2;

			line += separator + label;

			dataWriter.println(line);
		}

	}

	private String fine_type(Mention m){
		if (m.mentionType == MentionType.NOMINAL){
			if (Pattern.matches("^(the|this|that|these|those|my|your|his|her|its|our|their)", m.originalSpan.get(0).word().toLowerCase()))
				return "DEF";
			if (m.headWord.get(CoreAnnotations.PartOfSpeechAnnotation.class).startsWith("NNP"))
				return "DEF";
			else return "INDEF";
		}
		else if (m.mentionType == MentionType.PRONOMINAL){
			String pronoun = m.originalSpan.get(0).word().toLowerCase();
			if (Pattern.matches("^(he|him|himself|his)$",  pronoun))
				return "he";
			else if (Pattern.matches("^(she|her|herself|hers|her)$",  pronoun))
				return "she";
			else if (Pattern.matches("^(it|itself|its)$",  pronoun))
				return "it";
			else if (Pattern.matches("^(they|them|themselves|theirs|their)$",  pronoun))
				return "they";
			else if (Pattern.matches("^(i|me|myself|mine|my)$",  pronoun))
				return "i";
			else if (Pattern.matches("^(you|yourself|yourselves|yours|your)$",  pronoun))
				return "you";
			else if (Pattern.matches("^(we|us|ourselves|ours|our)$",  pronoun))
				return "we";
		}
		else if (m.mentionType == MentionType.PROPER)
			return "NAM";

		return "Other";

	}

	@Override
	public void finish() throws Exception {
		dataWriter.close();
	}

	public static void exportData(String outputPath, Dataset dataset, Properties props,
			Dictionaries dictionaries, Map<String, Map<Integer, Double>> anaphoricityScores) throws Exception {
		CorefProperties.setInput(props, dataset);
		String dataPath = outputPath + "/data_anaphoricity/";
		IOUtils.ensureDir(new File(outputPath));
		new AnaphoricityDataGenerator(props, dictionaries,
				dataPath + dataset.toString().toLowerCase(), anaphoricityScores).run(props, dictionaries);
	}

	public static void main(String[] args) throws Exception {
		String ana_path = "/data/nlp/moosavne/git/CoreNLP/data_anaphoricity/deep_coref_svm/";
		String sing_path = "/data/nlp/moosavne/git/CoreNLP/data_singleton/deep_coref_svm/";
		String[] svmOutFileNames= {ana_path+"train_scores",ana_path+"dev_scores",ana_path+"test_scores",
				sing_path+"train_scores",sing_path+"dev_scores",sing_path+"test_scores"};
		String[] svmMentionIds = {ana_path+"train_ids",ana_path+"dev_ids",ana_path+"test_ids",
				sing_path+"train_ids",sing_path+"dev_ids",sing_path+"test_ids"};
		
		
		List<Map<String, Map<Integer, Double>>> allDatasetAnaphoricityScores = new ArrayList<>();
//		List<Map<String, Map<Integer, Double>>> allDatasetSingletonScores = new ArrayList<>();

		
		for (int i = 0; i < 3; i++){
			List<String> outLines =  Files.readAllLines(Paths.get(svmOutFileNames[i]));
			List<String> idLines =  Files.readAllLines(Paths.get(svmMentionIds[i]));
			
			Map<String, Map<Integer, Double>> idToScore = new HashMap<>();
			
			for (int k = 0; k < outLines.size();k++){
				double score = (Double.parseDouble(outLines.get(k)));
				String[] s = idLines.get(k).trim().split(" QQQQQQQQQQ ");
				if (!idToScore.containsKey(s[0]))
					idToScore.put(s[0], new HashMap<>());
				idToScore.get(s[0]).put(Integer.parseInt(s[1]), score);
				//String[] sLine = line.split("\\s+");
				//Integer.parseInt(sLine[1]), (int)Math.round(Double.parseDouble(sLine[3].substring(0, sLine[3].indexOf("]")))* 10)
			}
			
			if (i < 3)
				allDatasetAnaphoricityScores.add(idToScore);
//			else
//				allDatasetSingletonScores.add(idToScore);
		}
		
		Properties props = StringUtils.argsToProperties(new String[] {"-props", args[0]});
		Dictionaries dictionaries = new Dictionaries(props);
		String outputPath = args[1];
		exportData(outputPath, Dataset.TRAIN, props, dictionaries, allDatasetAnaphoricityScores.get(0));
		exportData(outputPath, Dataset.DEV, props, dictionaries, allDatasetAnaphoricityScores.get(1));
		exportData(outputPath, Dataset.TEST, props, dictionaries, allDatasetAnaphoricityScores.get(2));
	}
}
