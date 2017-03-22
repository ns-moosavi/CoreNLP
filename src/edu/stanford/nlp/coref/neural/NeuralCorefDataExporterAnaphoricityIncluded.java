package edu.stanford.nlp.coref.neural;

import java.io.File;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import javax.json.Json;
import javax.json.JsonArray;
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
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;

public class NeuralCorefDataExporterAnaphoricityIncluded implements CorefDocumentProcessor{
	private final boolean conll;
	private final PrintWriter dataWriter;
	private final PrintWriter goldClusterWriter;
	private final Dictionaries dictionaries;
	private final boolean useAnaphoricityScore = false;
	private final boolean useSingletonScore = false;
	private final boolean isTrain;
	private final boolean discardNegatives = true;
	private final double discard_ratio = 10;
	private final Map<String, Map<Integer, Double>> anaScores;
	private final Map<String, Map<Integer, Double>> singScores;
	int discardedNegative = 0, discardedPositive = 0;
	int allNegPairs = 0;
	int discardMentions = 0;
	int allMentions = 0;


	public NeuralCorefDataExporterAnaphoricityIncluded(Properties props, Dictionaries dictionaries, String dataPath,
			String goldClusterPath, Map<String, Map<Integer, Double>> anaphoricityScores, Map<String, Map<Integer, Double>> singScores, boolean isTrain) {
		this.anaScores = anaphoricityScores;
		this.singScores = singScores;
		this.isTrain = isTrain;

		conll = CorefProperties.conll(props);

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
		List<Integer> allGoldMentionIds = new ArrayList<Integer>();
		JsonArrayBuilder clusters = Json.createArrayBuilder();
		for (CorefCluster gold : document.goldCorefClusters.values()) {
			JsonArrayBuilder c = Json.createArrayBuilder();
			for (Mention m : gold.corefMentions) {
				allGoldMentionIds.add(m.mentionID);
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

		JsonObjectBuilder docFeatures = Json.createObjectBuilder();
		docFeatures.add("doc_id", id);
		docFeatures.add("type", document.docType == DocType.ARTICLE ? 1 : 0);
		docFeatures.add("source", document.docInfo.get("DOC_ID").split("/")[0]);

		JsonArrayBuilder sentences = Json.createArrayBuilder();
		for (CoreMap sentence : document.annotation.get(SentencesAnnotation.class)) {
			sentences.add(getSentenceArray(sentence.get(CoreAnnotations.TokensAnnotation.class)));
		}

		List<Integer> discardedMentionIds = new ArrayList<Integer>();

		JsonObjectBuilder mentions = Json.createObjectBuilder();
		int singletonMentions = 0;
		for (Mention m : document.predictedMentionsByID.values()) {
			allMentions++;
			if (!document.goldMentionsByID.containsKey(m.mentionID))
				singletonMentions++;

			if (discardNegatives && isTrain && !document.goldMentionsByID.containsKey(m.mentionID) && singletonMentions % discard_ratio == 0){ 
				//					singScores.get(document.conllDoc.documentID+ document.conllDoc.getPartNo()).get(m.mentionID) <= -1){
				discardedMentionIds.add(m.mentionID);
			}
			else {
				Iterator<SemanticGraphEdge> iterator =
						m.enhancedDependency.incomingEdgeIterator(m.headIndexedWord);
				SemanticGraphEdge relation = iterator.hasNext() ? iterator.next() : null;
				String depRelation = relation == null ? "no-parent" : relation.getRelation().toString();
				String depParent = relation == null ? "<missing>" : relation.getSource().word();

				mentions.add(String.valueOf(m.mentionNum), Json.createObjectBuilder()
						.add("doc_id", id)
						.add("mention_id", m.mentionID)
						.add("mention_num", m.mentionNum)
						.add("sent_num", m.sentNum)
						.add("start_index", m.startIndex)
						.add("end_index", m.endIndex)
						.add("head_index", m.headIndex)
						.add("mention_type", m.mentionType.toString())
						.add("dep_relation", depRelation)
						.add("dep_parent", depParent)
						.add("sentence", getSentenceArray(m.sentenceWords))
						.add("contained-in-other-mention", mentionsByHeadIndex.get(m.headIndex).stream()
								.anyMatch(m2 -> m != m2 && m.insideIn(m2)) ? 1 : 0)
								//					.add("ana-score", (anaScores.get(document.conllDoc.documentID+ document.conllDoc.getPartNo()).get(m.mentionID)))
								//					.add("anaphoricity-score", anaphoricityScores.get(document.conllDoc.documentID).get(m.mentionID))
								.build());
			}
		}

		JsonArrayBuilder featureNames;

		featureNames = Json.createArrayBuilder()
				.add("same-speaker")
				.add("antecedent-is-mention-speaker")
				.add("mention-is-antecedent-speaker")
				.add("relaxed-head-match")
				.add("exact-string-match")
				.add("relaxed-string-match");


		if (useAnaphoricityScore){
			for (int i = 0; i < 6; i++)
				featureNames.add("ana-anaph-"+i);
		}
		if (useSingletonScore){
			for (int i = 0; i < 6; i++)
				featureNames.add("ante-coref-"+i);
		}
		JsonObjectBuilder features = Json.createObjectBuilder();
		JsonObjectBuilder labels = Json.createObjectBuilder();

		for (Map.Entry<Pair<Integer, Integer>, Boolean> e : mentionPairs.entrySet()) {
			Mention m1 = document.predictedMentionsByID.get(e.getKey().first);
			Mention m2 = document.predictedMentionsByID.get(e.getKey().second);
			String key = m1.mentionNum + " " + m2.mentionNum;

			if (!discardedMentionIds.contains(m1.mentionID) && !discardedMentionIds.contains(m2.mentionID)){
				JsonArrayBuilder builder = Json.createArrayBuilder();
				for (int val : CategoricalFeatureExtractor.pairwiseFeatures(document, m1, m2, dictionaries, conll, false)) {
					builder.add(val);
				}
				if (useAnaphoricityScore){
					for (int val : encodeScore(anaScores.get(document.conllDoc.documentID+ document.conllDoc.getPartNo()).get(m2.mentionID)))
						builder.add(val);
				}
				if (useSingletonScore){
					for (int val : encodeScore(singScores.get(document.conllDoc.documentID+ document.conllDoc.getPartNo()).get(m1.mentionID)))
						builder.add(val);
				}
				if (!e.getValue())
					allNegPairs++;

				features.add(key, builder.build());
				labels.add(key, e.getValue() ? 1 : 0);
			}
		}			



		JsonObject docData = Json.createObjectBuilder()
				.add("sentences", sentences.build())
				.add("mentions", mentions.build())
				.add("labels", labels.build())
				.add("pair_feature_names", featureNames.build())
				.add("pair_features", features.build())
				.add("document_features", docFeatures.build())
				.build();
		dataWriter.println(docData);
	}

	private int[] encodeScore(double score) {
		int[] m = new int[6];
		if (score <= -1)
			m[0] = 1;
		else if (score <=-0.5)
			m[1] = 1;
		else if (score < 0)
			m[2] = 1;
		else if (score >= 1)
			m[5] = 1;
		else if (score >= 0.5)
			m[4] = 1;
		else
			m[3] = 1;

		return m;
	}

	@Override
	public void finish() throws Exception {
		dataWriter.close();
		goldClusterWriter.close();
	}

	private static JsonArray getSentenceArray(List<CoreLabel> sentence) {
		JsonArrayBuilder sentenceBuilder = Json.createArrayBuilder();
		sentence.stream().map(CoreLabel::word)
		.map(w -> w.equals("/.") ? "." : w)
		.map(w -> w.equals("/?") ? "?" : w)
		.forEach(sentenceBuilder::add);
		return sentenceBuilder.build();
	}

	public static void exportData(String outputPath, Dataset dataset, Properties props,
			Dictionaries dictionaries, Map<String, Map<Integer, Double>> anaphoricityScores, 
			Map<String, Map<Integer, Double>> singScores, boolean isTrain) throws Exception {
		CorefProperties.setInput(props, dataset);
		String dataPath = outputPath + "/data_raw_ana/";
		String goldClusterPath = outputPath + "/gold_ana/";
		IOUtils.ensureDir(new File(outputPath));
		IOUtils.ensureDir(new File(dataPath));
		IOUtils.ensureDir(new File(goldClusterPath));
		NeuralCorefDataExporterAnaphoricityIncluded exporter = new NeuralCorefDataExporterAnaphoricityIncluded(props, dictionaries,
				dataPath + dataset.toString().toLowerCase(),
				goldClusterPath + dataset.toString().toLowerCase(), anaphoricityScores, singScores, isTrain);
		exporter.run(props, dictionaries);
		System.out.println(exporter.discardedPositive + " , " + exporter.discardedNegative + " from " + exporter.allNegPairs);
		System.out.println(exporter.discardMentions + " mention discarded from " + exporter.allMentions);
	}

	public static void main(String[] args) throws Exception {
		String ana_path = "/data/nlp/moosavne/git/CoreNLP/data_anaphoricity/deep_coref_svm/";
		String sing_path = "/data/nlp/moosavne/git/CoreNLP/data_singleton/deep_coref_svm/";
		String[] svmOutFileNames= {ana_path+"train_scores",ana_path+"dev_scores",ana_path+"test_scores",
				sing_path+"train_scores",sing_path+"dev_scores",sing_path+"test_scores"};
		String[] svmMentionIds = {ana_path+"train_ids",ana_path+"dev_ids",ana_path+"test_ids",
				sing_path+"train_ids",sing_path+"dev_ids",sing_path+"test_ids"};


		List<Map<String, Map<Integer, Double>>> allDatasetAnaphoricityScores = new ArrayList<>();
		List<Map<String, Map<Integer, Double>>> allDatasetSingletonScores = new ArrayList<>();


		for (int i = 0; i < 6; i++){
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
			else
				allDatasetSingletonScores.add(idToScore);
		}

		Properties props = StringUtils.argsToProperties(new String[] {"-props", args[0]});
		Dictionaries dictionaries = new Dictionaries(props);
		String outputPath = args[1];
		exportData(outputPath, Dataset.TRAIN, props, dictionaries, allDatasetAnaphoricityScores.get(0), allDatasetSingletonScores.get(0), true);
		exportData(outputPath, Dataset.DEV, props, dictionaries, allDatasetAnaphoricityScores.get(1), allDatasetSingletonScores.get(1), false);
		exportData(outputPath, Dataset.TEST, props, dictionaries, allDatasetAnaphoricityScores.get(2), allDatasetSingletonScores.get(2), false);
	}
}
