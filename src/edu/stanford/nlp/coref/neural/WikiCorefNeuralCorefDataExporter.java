package edu.stanford.nlp.coref.neural;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
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

import org.codehaus.jackson.JsonFactory;
import org.codehaus.jackson.JsonGenerator;

import edu.stanford.nlp.coref.CorefUtils;
import edu.stanford.nlp.coref.data.Dictionaries;
import edu.stanford.nlp.coref.data.Document;
import edu.stanford.nlp.coref.data.DocumentMaker;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.coref.data.Document.DocType;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

public class WikiCorefNeuralCorefDataExporter {
	public final PrintWriter dataWriter;
//	public final JsonGenerator dataWriter;
	JsonFactory f;
	private final Dictionaries dictionaries;
	private final PrintWriter goldClusterWriter;

	private CategoricalFeatureExtractor extractor;
	private final DocumentMaker docMaker;
	StanfordCoreNLP pipeline;
	private final boolean useExtendedPairwiseFeatures = false;
	private final boolean useExtendedPairwiseAttributes = false;




	public WikiCorefNeuralCorefDataExporter(Properties props, Dictionaries dictionaries, String dataPath, String goldClusterPath) {
		extractor = new CategoricalFeatureExtractor(props, dictionaries);
		this.dictionaries = dictionaries;
		pipeline = new StanfordCoreNLP(props);

		try {
			docMaker = new DocumentMaker(props, dictionaries);
			f = new JsonFactory();
			dataWriter = IOUtils.getPrintWriter(dataPath);
//			dataWriter = f.createJsonGenerator(IOUtils.getPrintWriter(dataPath));
			goldClusterWriter = IOUtils.getPrintWriter(goldClusterPath);

		} catch (Exception e) {
			throw new RuntimeException("Error creating data exporter", e);
		}
	}



	public void process(int id, String textFileName) {

		//Reading text file
		String tokenizedText = "", line = "";
		Document document;
		String docId;

		try {		
			BufferedReader br = new BufferedReader(new FileReader(textFileName));

			while ((line = br.readLine()) != null) {
				if (line.isEmpty())
					tokenizedText += "\n";
				else
					tokenizedText += line.trim() + " ";
			}
			br.close();
			docId = textFileName.substring(textFileName.lastIndexOf("/")+1);

			Annotation anno = new Annotation(tokenizedText);
			pipeline.annotate(anno);


			document = docMaker.makeDocument(anno);
		} catch (Exception e) {
			throw new RuntimeException("Error making document", e);
		}

		JsonArrayBuilder clusters = Json.createArrayBuilder();
		//		for (CorefCluster gold : document.goldCorefClusters.values()) {
		//			JsonArrayBuilder c = Json.createArrayBuilder();
		//			for (Mention m : gold.corefMentions) {
		//				c.add(m.mentionID);
		//			}
		//			clusters.add(c.build());
		//		}
		goldClusterWriter.println(Json.createObjectBuilder().add(String.valueOf(id),
				clusters.build()).build());


		Map<Pair<Integer, Integer>, Boolean> mentionPairs = CorefUtils.getUnlabeledMentionPairs(document);
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
		docFeatures.add("source", docId);

		JsonArrayBuilder sentences = Json.createArrayBuilder();
		for (CoreMap sentence : document.annotation.get(SentencesAnnotation.class)) {
			sentences.add(getSentenceArray(sentence.get(CoreAnnotations.TokensAnnotation.class)));
		}


		JsonObjectBuilder mentions = Json.createObjectBuilder();
		
		for (Mention m : document.predictedMentionsByID.values()) {
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
					.add("hear_ner", m.nerString)
					.add("gender", m.gender.toString())
					.add("number", m.number.toString())

					.add("dep_relation", depRelation)
					.add("dep_parent", depParent)
					.add("sentence", getSentenceArray(m.sentenceWords))
					.add("POSs", getPOSArray(m.sentenceWords))
					.add("contained-in-other-mention", mentionsByHeadIndex.get(m.headIndex).stream()
							.anyMatch(m2 -> m != m2 && m.insideIn(m2)) ? 1 : 0)
							.build());
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
				featureNames.add("new-"+c);

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
					.add("relaxed-string-match")
					.add("head-match")
					.add("ana-contains-ante")
					.add("ante-contains-ana")
					.add("ana-contains-ante-head")
					.add("ante-contains-ana-head");
		}

		JsonObjectBuilder features = Json.createObjectBuilder();
		JsonObjectBuilder labels = Json.createObjectBuilder();


//		JsonObject docData = Json.createObjectBuilder()
//				.add("sentences", sentences.build())
//				.add("mentions", mentions.build())
//				.add("labels", labels.build())
//				.add("pair_feature_names", featureNames.build())
//				.add("pair_features", features.build())
//				.add("document_features", docFeatures.build())
//				.build();
		
//		dataWriter.println(docData);
		dataWriter.print("{");
		dataWriter.print("\"sentences\":"+sentences.build()+",");
		dataWriter.print("\"mentions\":"+mentions.build()+",");
		dataWriter.print("\"pair_feature_names\":"+featureNames.build()+",");
		dataWriter.print("\"pair_features\":{");
		
		if (useExtendedPairwiseAttributes){
			Map<Integer, Map<Integer, List<Integer>>> allFeatures = 
					extractor.extendedPairwiseAttributes(document, mentionsList, dictionaries, false);
			int cnt = 0;
			for (Map.Entry<Pair<Integer, Integer>, Boolean> e : mentionPairs.entrySet()) {
				List<Integer> pairwiseFeatures = allFeatures.get(e.getKey().second).get(e.getKey().first);
				Mention m1 = document.predictedMentionsByID.get(e.getKey().first);
				Mention m2 = document.predictedMentionsByID.get(e.getKey().second);
				String key = m1.mentionNum + " " + m2.mentionNum;

				JsonArrayBuilder builder = Json.createArrayBuilder();
				for (int val : pairwiseFeatures) {
					builder.add(val);
				}
				dataWriter.print("\""+key+"\":"+ builder.build()+(cnt < mentionPairs.entrySet().size()-1 ? "," : ""));
//				features.add(key, builder.build());
				labels.add(key, e.getValue() ? 1 : 0);
				cnt++;
			}
		}
		else{
			int cnt = 0;
			for (Map.Entry<Pair<Integer, Integer>, Boolean> e : mentionPairs.entrySet()) {
				Mention m1 = document.predictedMentionsByID.get(e.getKey().first);
				Mention m2 = document.predictedMentionsByID.get(e.getKey().second);
				String key = m1.mentionNum + " " + m2.mentionNum;

				JsonArrayBuilder builder = Json.createArrayBuilder();
				List<Integer> pairFeatures = CategoricalFeatureExtractor.pairwiseFeatures(document, m1, m2, dictionaries, false, false);
				for (int val : pairFeatures) {
					builder.add(val);
				}
				dataWriter.print("\""+key+"\":"+ builder.build()+(cnt < (mentionPairs.entrySet().size()-1) ? "," : ""));

//				features.add(key, builder.build());
				labels.add(key, e.getValue() ? 1 : 0);
				cnt++;
			}	
		}
		dataWriter.print("},");
		dataWriter.print("\"labels\":"+labels.build()+",");
		dataWriter.print("\"document_features\":"+docFeatures.build());

		dataWriter.print("}");
		dataWriter.println();
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
	public static void exportData(String outputPath) throws Exception {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,mention,coref");
		props.setProperty("tokenize.whitespace", "true");
		props.setProperty("ssplit.eolonly", "true");
		props.setProperty("coref.algorithm", "neural");
		Dictionaries dictionaries = new Dictionaries(props);

		String dataPath = outputPath + "/data_raw/";
		String goldPath = outputPath + "/gold/";
		IOUtils.ensureDir(new File(outputPath));
		IOUtils.ensureDir(new File(dataPath));
		IOUtils.ensureDir(new File(goldPath));
		WikiCorefNeuralCorefDataExporter dataExporter = new WikiCorefNeuralCorefDataExporter(props, dictionaries,
				dataPath+"wikicoref", goldPath+"wikicoref");
		System.out.println(Arrays.toString(dataExporter.extractor.counts));
		File docFolder = new File("/data/nlp/moosavne/corpora/WikiCoref/Documents/");
		File[] listOfFiles = docFolder.listFiles();
		Arrays.sort(listOfFiles);
		for (int i = 0; i < listOfFiles.length; i++) {
			System.out.println(i);
			dataExporter.process(i, "/data/nlp/moosavne/corpora/WikiCoref/Annotation/"+listOfFiles[i].getName().replaceAll("\\s", "_")+"/"+listOfFiles[i].getName()+".txt");
		}

		dataExporter.dataWriter.close();

	}

	public static void main(String[] args) throws Exception {
		String outputPath = args[0];

		exportData(outputPath);

	}

}
