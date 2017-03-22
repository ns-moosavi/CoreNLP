package edu.stanford.nlp.coref.neural;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
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

import edu.stanford.nlp.coref.CorefUtils;
import edu.stanford.nlp.coref.data.CorefCluster;
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

public class WikiCorefMentionStat {
	public final PrintWriter dataWriter;
	private final Dictionaries dictionaries;
	private final PrintWriter goldClusterWriter;

	private CategoricalFeatureExtractor extractor;
	private final DocumentMaker docMaker;
	StanfordCoreNLP pipeline;
 	int properNum = 0;
    	int commonNum = 0;
	int pronounNum = 0;



	public WikiCorefMentionStat(Properties props, Dictionaries dictionaries, String dataPath, String goldClusterPath) {
		extractor = new CategoricalFeatureExtractor(props, dictionaries);
		this.dictionaries = dictionaries;
		pipeline = new StanfordCoreNLP(props);

		try {
			docMaker = new DocumentMaker(props, dictionaries);
			dataWriter = IOUtils.getPrintWriter(dataPath);
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

		for (Mention m : document.predictedMentionsByID.values()) {
			if (m.mentionType.toString().equals("PROPER"))
				properNum++;
			else if (m.mentionType.toString().equals("NOMINAL"))
				commonNum++;
			else if (m.mentionType.toString().equals("PRONOMINAL"))
				pronounNum++;
		}

	}



	public static void exportData(String outputPath) throws Exception {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,mention,coref");
		props.setProperty("tokenize.whitespace", "true");
		props.setProperty("ssplit.eolonly", "true");
		props.setProperty("coref.algorithm", "neural");
		Dictionaries dictionaries = new Dictionaries(props);

		String dataPath = outputPath + "/wikicoref_data_raw/";
		String goldPath = outputPath + "/wikicoref_gold/";
		IOUtils.ensureDir(new File(outputPath));
		IOUtils.ensureDir(new File(dataPath));
		WikiCorefMentionStat dataExporter = new WikiCorefMentionStat(props, dictionaries,
				dataPath+"wikicoref", goldPath+"wikicoref");
		System.out.println(Arrays.toString(dataExporter.extractor.counts));
		File docFolder = new File("/data/nlp/moosavne/corpora/WikiCoref/Documents/");
		File[] listOfFiles = docFolder.listFiles();

		for (int i = 0; i < listOfFiles.length; i++) {
			//			System.out.println("/data/nlp/moosavne/corpora/WikiCoref/Annotation/"+listOfFiles[i].getName().replaceAll("\\s", "_")+"/"+listOfFiles[i].getName()+".txt");
			dataExporter.process(i, "/data/nlp/moosavne/corpora/WikiCoref/Annotation/"+listOfFiles[i].getName().replaceAll("\\s", "_")+"/"+listOfFiles[i].getName()+".txt");
		}
		System.out.println("Proper: " + dataExporter.properNum + " Common: " + dataExporter.commonNum + " Pronoun: " + dataExporter.pronounNum);
		dataExporter.dataWriter.close();

	}

	public static void main(String[] args) throws Exception {
		String outputPath = args[0];

		exportData(outputPath);

	}

}
