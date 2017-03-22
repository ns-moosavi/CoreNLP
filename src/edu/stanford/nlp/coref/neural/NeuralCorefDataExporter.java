package edu.stanford.nlp.coref.neural;

import java.io.File;
import java.io.PrintWriter;
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
import edu.stanford.nlp.coref.CorefProperties.Dataset;
import edu.stanford.nlp.coref.CorefUtils;
import edu.stanford.nlp.coref.data.CorefCluster;
import edu.stanford.nlp.coref.data.Dictionaries;
import edu.stanford.nlp.coref.data.Document;
import edu.stanford.nlp.coref.data.Document.DocType;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;

/**
 * Outputs the CoNLL CoNLL data for training the neural coreference system
 * (implented in python/theano).
 * See <a href="https://github.com/clarkkev/deep-coref">https://github.com/clarkkev/deep-coref</a>
 * for the training code.
 * @author Kevin Clark
 */
public class NeuralCorefDataExporter implements CorefDocumentProcessor {
	private final boolean conll;
	private final PrintWriter dataWriter;
	private final PrintWriter goldClusterWriter;
	private final Dictionaries dictionaries;
	private final boolean useExtendedPairwiseFeatures = false;
	private final boolean useExtendedPairwiseAttributes = true;
	private CategoricalFeatureExtractor extractor;
	private final boolean useNAMFeatures = false;
	private final boolean discardNegatives = false;
	private final double discard_ratio = 4;
	private final boolean isTrain;


	public NeuralCorefDataExporter(Properties props, Dictionaries dictionaries, String dataPath,
			String goldClusterPath, boolean isTrain) {
		conll = CorefProperties.conll(props);
		extractor = new CategoricalFeatureExtractor(props, dictionaries);
		this.isTrain = isTrain;
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
		int singletonMentions = 0;

		JsonObjectBuilder mentions = Json.createObjectBuilder();
		for (Mention m : document.predictedMentionsByID.values()) {
			if (!allGoldMentionIds.contains(m.mentionID))
				singletonMentions++;

			if (discardNegatives && isTrain && !allGoldMentionIds.contains(m.mentionID) && singletonMentions % discard_ratio == 0){ 
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
						//					.add("hear_ner", m.nerString)
						//					.add("gender", m.gender.toString())
						//					.add("number", m.number.toString())
						.add("sentence", getSentenceArray(m.sentenceWords))
						//   					.add("POSs", getPOSArray(m.sentenceWords))
						.add("contained-in-other-mention", mentionsByHeadIndex.get(m.headIndex).stream()
								.anyMatch(m2 -> m != m2 && m.insideIn(m2)) ? 1 : 0)
								.build());
			}
		}

		JsonArrayBuilder featureNames;

		if (useExtendedPairwiseFeatures){
			int newFeatures = extractor.parser.m_allFeatures.size();


			featureNames = Json.createArrayBuilder()
					.add("same-speaker")
					.add("antecedent-is-mention-speaker")
					.add("mention-is-antecedent-speaker")
					.add("relaxed-head-match")
					.add("exact-string-match")
					.add("relaxed-string-match");
			for (int c = 0; c < newFeatures; c++)
				featureNames.add("n-"+c);
		}
		else if (useExtendedPairwiseAttributes){

			featureNames = Json.createArrayBuilder()
					.add("same-speaker")
					.add("antecedent-is-mention-speaker")
					.add("mention-is-antecedent-speaker")
					.add("relaxed-head-match")
					.add("exact-string-match")
					.add("relaxed-string-match");
			for (int c = 0; c < extractor.attribute_size; c++)
				featureNames.add("n"+c);

		}
		else{
			featureNames = Json.createArrayBuilder()
					.add("same-speaker")
					.add("antecedent-is-mention-speaker")
					.add("mention-is-antecedent-speaker")
					.add("relaxed-head-match")
					.add("exact-string-match")
					.add("relaxed-string-match");
		}
		JsonObjectBuilder features = Json.createObjectBuilder();
		JsonObjectBuilder labels = Json.createObjectBuilder();

		if (useExtendedPairwiseFeatures){
			Map<Integer, Map<Integer, List<Integer>>> allFeatures = 
					extractor.extendedPairwiseFeatures(document, mentionsList, dictionaries, conll);

			for (Map.Entry<Pair<Integer, Integer>, Boolean> e : mentionPairs.entrySet()) {

				List<Integer> pairwiseFeatures = allFeatures.get(e.getKey().second).get(e.getKey().first);
				//				System.out.println(pairwiseFeatures.size());
				Mention m1 = document.predictedMentionsByID.get(e.getKey().first);
				Mention m2 = document.predictedMentionsByID.get(e.getKey().second);
				String key = m1.mentionNum + " " + m2.mentionNum;
				if (!discardedMentionIds.contains(m1.mentionID) && !discardedMentionIds.contains(m2.mentionID)){

					JsonArrayBuilder builder = Json.createArrayBuilder();
					for (int val : pairwiseFeatures) {
						builder.add(val);
					}
					features.add(key, builder.build());
					labels.add(key, e.getValue() ? 1 : 0);
				}
			}
		}
		else if (useExtendedPairwiseAttributes){
			Map<Integer, Map<Integer, List<Integer>>> allFeatures = 
					extractor.extendedPairwiseAttributes(document, mentionsList, dictionaries, conll);

			for (Map.Entry<Pair<Integer, Integer>, Boolean> e : mentionPairs.entrySet()) {
				List<Integer> pairwiseFeatures = allFeatures.get(e.getKey().second).get(e.getKey().first);
				Mention m1 = document.predictedMentionsByID.get(e.getKey().first);
				Mention m2 = document.predictedMentionsByID.get(e.getKey().second);
				if (!discardedMentionIds.contains(m1.mentionID) && !discardedMentionIds.contains(m2.mentionID)){

					String key = m1.mentionNum + " " + m2.mentionNum;

					JsonArrayBuilder builder = Json.createArrayBuilder();
					for (int val : pairwiseFeatures) {
						builder.add(val);
					}
					features.add(key, builder.build());
					labels.add(key, e.getValue() ? 1 : 0);
				}
			}
		}
		else{
			for (Map.Entry<Pair<Integer, Integer>, Boolean> e : mentionPairs.entrySet()) {
				Mention m1 = document.predictedMentionsByID.get(e.getKey().first);
				Mention m2 = document.predictedMentionsByID.get(e.getKey().second);
				String key = m1.mentionNum + " " + m2.mentionNum;
				if (!discardedMentionIds.contains(m1.mentionID) && !discardedMentionIds.contains(m2.mentionID)){

					JsonArrayBuilder builder = Json.createArrayBuilder();
					List<Integer> pairFeatures = CategoricalFeatureExtractor.pairwiseFeatures(document, m1, m2, dictionaries, conll, useNAMFeatures);
					for (int val : pairFeatures) {
						builder.add(val);
					}
					features.add(key, builder.build());
					labels.add(key, e.getValue() ? 1 : 0);
				}
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

	private static JsonArray getPOSArray(List<CoreLabel> sentence) {
		JsonArrayBuilder builder = Json.createArrayBuilder();
		for (int i = 0; i < sentence.size(); i++){
			builder.add(sentence.get(i).get(CoreAnnotations.PartOfSpeechAnnotation.class));
		}
		return builder.build();
	}

	public static void exportData(String outputPath, Dataset dataset, Properties props,
			Dictionaries dictionaries, boolean isTrain) throws Exception {
		CorefProperties.setInput(props, dataset);
		String dataPath = outputPath + ( "/data_raw_extended/");
		String goldClusterPath = outputPath + ("/gold_extended/");
		IOUtils.ensureDir(new File(outputPath));
		IOUtils.ensureDir(new File(dataPath));
		IOUtils.ensureDir(new File(goldClusterPath));
		NeuralCorefDataExporter dataExporter = new NeuralCorefDataExporter(props, dictionaries,
				dataPath + dataset.toString().toLowerCase(),
				goldClusterPath + dataset.toString().toLowerCase(), isTrain);

		dataExporter.run(props, dictionaries);


		return;
	}

	public static void main(String[] args) throws Exception {
		Properties props = StringUtils.argsToProperties(new String[] {"-props", args[0]});
		Dictionaries dictionaries = new Dictionaries(props);
		String outputPath = args[1];
		//		int[] dev_feat_count = exportData(outputPath, Dataset.DEV, props, dictionaries, null);

		exportData(outputPath, Dataset.TRAIN, props, dictionaries, true);
		exportData(outputPath, Dataset.DEV, props, dictionaries, false);
		//		System.out.println(dev_feat_count.length);
		exportData(outputPath, Dataset.TEST, props, dictionaries, false);
		//		System.out.println(dev_feat_count.length);
		//		System.out.println();
	}

}
