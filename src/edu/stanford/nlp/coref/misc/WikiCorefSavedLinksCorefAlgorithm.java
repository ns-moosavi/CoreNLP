package edu.stanford.nlp.coref.misc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.io.InputStreamReader;
import java.io.FileInputStream;

import edu.stanford.nlp.coref.data.CorefCluster;
import edu.stanford.nlp.coref.data.Dictionaries;
import edu.stanford.nlp.coref.data.Document;
import edu.stanford.nlp.coref.data.DocumentMaker;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

public class WikiCorefSavedLinksCorefAlgorithm {
	private PrintWriter dataWriter;
	private StanfordCoreNLP pipeline;
	private DocumentMaker docMaker;
	PrintWriter corr_pw;
	private final Map<Integer, List<Pair<Integer, Integer>>> toMerge = new HashMap<>();

	public WikiCorefSavedLinksCorefAlgorithm(String dataPath, String savedLinkPath){
		try {
			corr_pw = new PrintWriter("wikicoref_made_decisions.txt");
			dataWriter = new PrintWriter(dataPath+".out");
			Properties props = new Properties();
			props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,mention,coref");
			props.setProperty("tokenize.whitespace", "true");
			props.setProperty("ssplit.eolonly", "true");
			props.setProperty("coref.algorithm", "neural");
			Dictionaries dictionaries = new Dictionaries(props);
			pipeline = new StanfordCoreNLP(props);

			docMaker = new DocumentMaker(props, dictionaries);


		} catch (Exception e) {
			e.printStackTrace();
		}
		try(BufferedReader br = new BufferedReader(new FileReader(savedLinkPath))) {
			br.lines().forEach(line -> {
				String[] split = line.split("\t");
				int did = Integer.valueOf(split[0]);

				List<Pair<Integer, Integer>> docMerges = toMerge.get(did);
				if (docMerges == null) {
					docMerges = new ArrayList<>();
					toMerge.put(did, docMerges);
				}

				if (split.length > 1) {
					String[] pairs = split[1].split(" ");
					for (String pair : pairs) {
						String[] ms = pair.split(",");
						docMerges.add(new Pair<>(Integer.valueOf(ms[0]), Integer.valueOf(ms[1])));
					}
				}
			});
		} catch (IOException e) {
			throw new RuntimeException("Error reading saved links", e);
		}
	}

	void mergeCoreferenceClusters(Pair<Integer, Integer> mentionPair,
			Document document) {
		Mention m1 = document.predictedMentionsByID.get(mentionPair.first);
		Mention m2 = document.predictedMentionsByID.get(mentionPair.second);
//		if (m1 == null){
//			System.out.println(mentionPair.first + " " + mentionPair.second + " " + document.predictedMentionsByID.size());
//			System.out.println(Arrays.toString(document.predictedMentionsByID.keySet().toArray()));
//		}
//		if (m2 == null){
//			System.out.println("2 " + mentionPair.first + " " + mentionPair.second + " " + document.predictedMentionsByID.size());
//			for (Integer ii : document.predictedMentionsByID.keySet())
//				System.out.println(ii + " " + document.predictedMentionsByID.get(ii));
//		}
		int removeId = m1.corefClusterID;
		CorefCluster c1= null, c2=null;
		if (document.corefClusters.containsKey(m1.corefClusterID))
			c1 = document.corefClusters.get(m1.corefClusterID);
		if (document.corefClusters.containsKey(m2.corefClusterID))
			c2 = document.corefClusters.get(m2.corefClusterID);
		if (c1 != null && c2 != null){
			CorefCluster.mergeClusters(c2, c1);
	        if (corr_pw != null){
                boolean are_coref = false;
//			if(document.goldMentionsByID.containsKey(m1.mentionID) && document.goldMentionsByID.containsKey(m2.mentionID) &&
//				document.goldMentionsByID.get(m1.mentionID).goldCorefClusterID == document.goldMentionsByID.get(m2.mentionID).goldCorefClusterID)
				are_coref = true;
	        	if (!m1.headString.equalsIgnoreCase(m2.headString))
	        		corr_pw.println((are_coref ? 1 : 0) + "\t"+m1.mentionType + "\t" + m2.mentionType+"\t"+m1.spanToString() + " " + m2.spanToString() + "\t" + m1.headString + " " + m2.headString + "\t"+ "-" + " " + "-");
	        }
		}
		if (document.corefClusters.containsKey(removeId))
			document.corefClusters.remove(removeId);
	}

	public void parseDocument(int id, String textFile){

		try {
			if (toMerge.containsKey(id)) {
				String tokenizedText = "", line = "";
				Document document;
				String docId;

				BufferedReader br = new BufferedReader(new FileReader(textFile));

				while ((line = br.readLine()) != null) {
					if (line.isEmpty())
						tokenizedText += "\n";
					else
						tokenizedText += line.trim() + " ";
				}
				br.close();
				docId = textFile.substring(textFile.lastIndexOf("/")+1, textFile.indexOf(".txt"));

				Annotation anno = new Annotation(tokenizedText);
				pipeline.annotate(anno);


				document = docMaker.makeDocument(anno);

				System.out.println(id);
				for (Pair<Integer, Integer> pair : toMerge.get(id)) {
					mergeCoreferenceClusters(pair, document);
				}
				dataWriter.println("#begin document "+docId);
				Map<Integer, Map<Integer, String>> mentionIndexToChainId = new HashMap<Integer, Map<Integer,String>>();
				for (Integer chainId: document.corefClusters.keySet()){
					if (document.corefClusters.get(chainId).size() > 1){
						for (Mention men : document.corefClusters.get(chainId).getCorefMentions()){

							if(!mentionIndexToChainId.containsKey(men.sentNum))
								mentionIndexToChainId.put(men.sentNum, new HashMap<>());

							if (men.endIndex-men.startIndex==1){
								if (mentionIndexToChainId.get(men.sentNum).containsKey(men.startIndex))
									mentionIndexToChainId.get(men.sentNum).put(men.startIndex, mentionIndexToChainId.get(men.sentNum).get(men.startIndex)+"|("+chainId+")");
								else
									mentionIndexToChainId.get(men.sentNum).put(men.startIndex,"("+chainId+")");
							}
							else {
								if (mentionIndexToChainId.get(men.sentNum).containsKey(men.startIndex))
									mentionIndexToChainId.get(men.sentNum).put(men.startIndex, mentionIndexToChainId.get(men.sentNum).get(men.startIndex)+"|("+chainId);
								else
									mentionIndexToChainId.get(men.sentNum).put(men.startIndex,"("+chainId);

								if (mentionIndexToChainId.get(men.sentNum).containsKey(men.endIndex-1))
									mentionIndexToChainId.get(men.sentNum).put(men.endIndex-1, mentionIndexToChainId.get(men.sentNum).get(men.endIndex-1)+"|"+chainId+")");
								else
									mentionIndexToChainId.get(men.sentNum).put(men.endIndex-1,chainId+")");
							}

						}
					}
				}
				List<CoreMap> sentences = anno.get(CoreAnnotations.SentencesAnnotation.class);

				for (int j = 0; j < sentences.size();j++){
					if (mentionIndexToChainId.containsKey(j)){
						for (int k = 0; k < sentences.get(j).get(CoreAnnotations.TokensAnnotation.class).size();k++)
							if (mentionIndexToChainId.get(j).containsKey(k))
								dataWriter.println("Nan\t0\t"+(k+1)+"\t"+ sentences.get(j).get(CoreAnnotations.TokensAnnotation.class).get(k).originalText()+"\t"+mentionIndexToChainId.get(j).get(k));
							else
								dataWriter.println("Nan\t0\t"+(k+1)+"\t"+ sentences.get(j).get(CoreAnnotations.TokensAnnotation.class).get(k).originalText()+"\t-");
					}
					else{
						for (int k = 0; k < sentences.get(j).get(CoreAnnotations.TokensAnnotation.class).size();k++)
							dataWriter.println("Nan\t0\t"+(k+1)+"\t"+ sentences.get(j).get(CoreAnnotations.TokensAnnotation.class).get(k).originalText()+"\t-");
					}
					dataWriter.println();
				}
				dataWriter.println("#end document");

			}
		}catch (Exception err) {
			err.printStackTrace();
			System.out.println(err.getMessage());

		}

		return; 
	}
	public static void main(String[] args){
		WikiCorefSavedLinksCorefAlgorithm nn = new WikiCorefSavedLinksCorefAlgorithm(args[0], args[1]);
		File docFolder = new File("/data/nlp/moosavne/corpora/WikiCoref/Documents/");
		File[] listOfFiles = docFolder.listFiles();
		
		Arrays.sort(listOfFiles);
		for (int i = 0; i < listOfFiles.length; i++) {
			System.out.println("/data/nlp/moosavne/corpora/WikiCoref/Annotation/"+listOfFiles[i].getName().replaceAll("\\s", "_")+"/"+listOfFiles[i].getName()+".txt");
			nn.parseDocument(i, "/data/nlp/moosavne/corpora/WikiCoref/Annotation/"+listOfFiles[i].getName().replaceAll("\\s", "_")+"/"+listOfFiles[i].getName()+".txt");
		}
		nn.dataWriter.close();
		if (nn.corr_pw != null)
			nn.corr_pw.close();

	}
}
