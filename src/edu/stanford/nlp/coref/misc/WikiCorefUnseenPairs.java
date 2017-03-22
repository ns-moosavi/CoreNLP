package edu.stanford.nlp.coref.misc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.CorefProperties;
import edu.stanford.nlp.coref.CorefProperties.Dataset;
import edu.stanford.nlp.coref.data.Dictionaries;
import edu.stanford.nlp.coref.data.Dictionaries.MentionType;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;

public class WikiCorefUnseenPairs {

	int non_coref_mentions = 0, coref_mentions = 0, seen_as_coref = 0, seen_as_non_coref = 0;
	StanfordCoreNLP pipeline;

	public WikiCorefUnseenPairs() {
		Properties wiki_props = new Properties();
		wiki_props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,mention,coref");
		wiki_props.setProperty("tokenize.whitespace", "true");
		wiki_props.setProperty("ssplit.eolonly", "true");
		wiki_props.setProperty("coref.algorithm", "neural");
		pipeline = new StanfordCoreNLP(wiki_props);	
	}

	public void parseDocument(String textFile,String markableXMLFile, Set<String> trainGoldMentions, Set<String> trainNonCorefMentions){
		Map<Integer, List<Integer>> docAnaphoricMentionSpans = new HashMap<>();

		try {
			//Reading text file
			String tokenizedText = "", line = "";

			BufferedReader br = new BufferedReader(new FileReader(textFile));

			while ((line = br.readLine()) != null) {
				if (line.isEmpty())
					tokenizedText += "\n";
				else
					tokenizedText += line.trim() + " ";
			}
			br.close();
			String docId = textFile.substring(textFile.lastIndexOf("/")+1);

			Annotation document = new Annotation(tokenizedText);
			pipeline.annotate(document);

			System.out.println("---");
			//		    System.out.println("coref chains");
			//		    for (CorefChain cc : document.get(CorefCoreAnnotations.CorefChainAnnotation.class).values()) {
			//		      System.out.println("\t" + cc);
			//		    }
			int wordIndex = 1;

			Map<Integer, List<Pair<Integer,Integer>>> sentSysMarkableInd = new HashMap<>();
			Map<Pair<Integer, Pair<Integer,Integer>>,Mention> indexToMention = new HashMap<>();
			Map<Integer, Map<Integer, Pair<Integer, Integer>>> sentWordtoIndex = new HashMap<>();
			Map<Integer, Pair<Integer, Integer>> wordSentIndex = new HashMap<>();
			List<Mention> allSysMentions = new ArrayList<>();
			//			PrintWriter pw = new PrintWriter(textFile+".temp.txt");

			List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
			int sentNum = 0;
			int sysMenNum = 0;
			for (CoreMap sentence : sentences) {
				sentSysMarkableInd.put(sentNum, new ArrayList<>());
				sentWordtoIndex.put(sentNum, new HashMap<>());

				int sentIndex = 0;

				for (Mention m : sentence.get(CorefCoreAnnotations.CorefMentionsAnnotation.class)) {
					sentSysMarkableInd.get(m.sentNum).add(new Pair<>(m.startIndex, m.endIndex-1));
					Pair<Integer, Pair<Integer,Integer>> index = new Pair<>(m.sentNum, new Pair<>(m.startIndex, m.endIndex-1));
					indexToMention.put(index, m);
					allSysMentions.add(m);
					sysMenNum++;
				}
				//				String prevWord = "";
				for (CoreMap token: sentence.get(CoreAnnotations.TokensAnnotation.class)) {
					Pair<Integer, Integer> p = new Pair<Integer, Integer>(sentNum, sentIndex);
					wordSentIndex.put(wordIndex, p);
					//					String tokenText = token.get(CoreAnnotations.TextAnnotation.class);
					//					pw.println(tokenText);
					//					if (tokenText.equals("I.") || tokenText.equals("°C") || tokenText.equals("°F"))
					//						wordIndex++;
					//					if ((prevWord.equals("Bros.") || prevWord.equals("Jr.")) && tokenText.equals("."))
					//						continue;
					wordIndex++;
					sentIndex++;
					//					prevWord = tokenText;
				}
				//				pw.println("");
				sentNum++;
			}
			System.out.println(sysMenNum + " System mention detected");
			int gold_mentions = 0;
			int detected_gold_mentions = 0;
			Set<String> metCorefClusters = new HashSet<>();
			List<Mention> goldMentions = new ArrayList<>();

			//Readng XMl file
			DocumentBuilderFactory docBuilderFactory = DocumentBuilderFactory.newInstance();
			DocumentBuilder docBuilder = docBuilderFactory.newDocumentBuilder();
			org.w3c.dom.Document doc = docBuilder.parse (new File(markableXMLFile));

			doc.getDocumentElement ().normalize ();

			NodeList listOfMarkable = doc.getElementsByTagName("markable");

			for(int s=0; s<listOfMarkable.getLength() ; s++){

				Node markable = listOfMarkable.item(s);
				if(markable.getNodeType() == Node.ELEMENT_NODE){

					Element curr = (Element)markable;
					boolean includeMention = curr.getAttribute("coreftype").equals("ident");
					if (includeMention){
						gold_mentions++;
						String corefID = curr.getAttribute("coref_class");
						String spanStr = curr.getAttribute("span");
						int spanStart = Integer.parseInt(spanStr.substring(5, spanStr.indexOf("..")));
						int spanEnd = Integer.parseInt(spanStr.substring(spanStr.indexOf("..")+7));
						if (!docAnaphoricMentionSpans.containsKey(spanStart))
							docAnaphoricMentionSpans.put(spanStart, new ArrayList<>());

						docAnaphoricMentionSpans.get(spanStart).add(spanEnd);

						Pair<Integer, Integer> fIndices = wordSentIndex.get(spanStart);
						Pair<Integer, Integer> lIndices = wordSentIndex.get(spanEnd);
						if ((int) fIndices.first != (int)lIndices.first){
							System.out.println("Should not happen: " + textFile + " " + fIndices + " -- " + lIndices + " " + spanStr);
							continue;

						}
						int gMenSentNum = fIndices.first();
						Pair<Integer, Integer> p = new Pair<>(fIndices.second(), lIndices.second());
						Pair<Integer, Pair<Integer,Integer>> index = new Pair<>(gMenSentNum, p);

						if (sentSysMarkableInd.containsKey(gMenSentNum)){
							if (sentSysMarkableInd.get(gMenSentNum).contains(p)){
								detected_gold_mentions++;
								Mention m = indexToMention.get(index);
								if (m.mentionType != MentionType.PRONOMINAL){
									if (trainGoldMentions.contains(m.spanToString().toLowerCase()))
										seen_as_coref++;
									else if (trainNonCorefMentions.contains(m.spanToString().toLowerCase()))
										seen_as_non_coref++;
								}

							}
							//							else
							//								System.out.println(spanStr + " not detected");
						}
					}
				}
			}
			coref_mentions += detected_gold_mentions;
			non_coref_mentions += (allSysMentions.size() - detected_gold_mentions);

			System.out.println("Gold mentions: " + gold_mentions + " detected gold mentions: " + detected_gold_mentions);

		}catch (Exception err) {
			System.out.println(err.getMessage());

		}

		return; 
	}

	public static void main(String[] args) throws Exception {
		Properties props = StringUtils.argsToProperties(new String[] {"-props", args[0]});
		Dictionaries dictionaries = new Dictionaries(props);
		CorefProperties.setInput(props, Dataset.TRAIN);
		TrainMentionCollector collector = new TrainMentionCollector();
		collector.run(props, dictionaries);

		WikiCorefUnseenPairs wikiCorefCounter = new WikiCorefUnseenPairs();
		TrainMentionCollector trainMentionCollector = new TrainMentionCollector();
		trainMentionCollector.run(props, dictionaries);

		File docFolder = new File("/data/nlp/moosavne/corpora/WikiCoref/Documents/");
		File[] listOfFiles = docFolder.listFiles();
		Arrays.sort(listOfFiles);
		for (int i = 0; i < listOfFiles.length; i++) {
			wikiCorefCounter.parseDocument("/data/nlp/moosavne/corpora/WikiCoref/Annotation/"+listOfFiles[i].getName().replaceAll("\\s", "_")+"/"+listOfFiles[i].getName()+".txt", 
					"/data/nlp/moosavne/corpora/WikiCoref/Annotation/"+listOfFiles[i].getName().replaceAll("\\s", "_")+"/Markables/"+listOfFiles[i].getName()+"_coref_level_OntoNotesScheme.xml",
					trainMentionCollector.allGoldMentionStr, trainMentionCollector.allMentionStr);

		}
		System.out.println("All coref mentions: " + wikiCorefCounter.coref_mentions);
		System.out.println("All non-coreferent: " + wikiCorefCounter.non_coref_mentions);
		double coref_seen_ratio = (double) wikiCorefCounter.seen_as_coref/(double)wikiCorefCounter.coref_mentions;
		double seen_ratio = (double) (wikiCorefCounter.seen_as_coref+wikiCorefCounter.seen_as_non_coref) /(double) wikiCorefCounter.coref_mentions;
		System.out.println("seen as coref ratio: " + coref_seen_ratio);
		System.out.println("seen ratio: " + seen_ratio);
		System.out.println(wikiCorefCounter.seen_as_coref + " -- " + wikiCorefCounter.seen_as_non_coref);

	}
}
