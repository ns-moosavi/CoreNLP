package edu.stanford.nlp.coref;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import edu.stanford.nlp.coref.CorefProperties.Dataset;
import edu.stanford.nlp.coref.data.CorefCluster;
import edu.stanford.nlp.coref.data.Dictionaries;
import edu.stanford.nlp.coref.data.Document;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;

public class GenerateSVMData {
	static FeatureFlags flags;
	final boolean m_wikiCoref = true;
	final boolean m_anaphoricity = true;
	final boolean m_splitTrain = false;

	public GenerateSVMData(Properties props){
		flags = new FeatureFlags(props);
		flags.useLemma=true;
		flags.useHead=true;
		flags.useLowerCasedStrings=false;
		flags.useTags=true;
		flags.windowSize=2;
		flags.prunThreshold=10;
		flags.useLength=true;
		flags.useText=false;
		flags.useLinguistic=false;
		flags.usePosition=false;
		flags.useVerbs=false;
		flags.useNGrams=false;
		flags.lowercaseNGrams=false;
		flags.dehyphenateNGrams=false;
		flags.noMidNGrams=true;
		flags.maxNGramLeng=6;
		flags.useNPType=true;
		flags.useTextMatch=true;
		flags.useHeadMatch=true;
		flags.useWholeNPText = true;
		flags.useNewInnerWordRepresentation = true;

	}

	public static void main(String[] args){
		Properties props = StringUtils.argsToProperties(new String[] {"-props", args[0]});
		Dictionaries dictionaries;
		try {
			dictionaries = new Dictionaries(props);

			GenerateSVMData gsd = new GenerateSVMData(props);
			System.out.println("Generating training data");
			CorefProperties.setInput(props, Dataset.TRAIN);
			String dataPath = gsd.m_anaphoricity ? "./data_anaphoricity/" : "./data_singleton/";
			CoNLLDataGenerator trainGenerator = gsd.new CoNLLDataGenerator(gsd.m_splitTrain, false);
			trainGenerator.run(props, dictionaries);
			FeatureStruct trainFeatureInfo = gsd.new FeatureStruct();
			trainFeatureInfo.extractedPositiveFeatures = trainGenerator.extractedPositiveFeatures;
			trainFeatureInfo.extractedNegativeFeatures = trainGenerator.extractedNegativeFeatures;

			for (String feat : trainGenerator.discPosFeatures.keySet()){
				if (trainGenerator.discPosFeatures.get(feat) > trainGenerator.prunThreshold){
					//					System.out.print(trainGenerator.discPosFeatures.get(feat) + " " );
					trainGenerator.allDisFeatures.add(feat);
				}
			}
			trainFeatureInfo.allDisFeatures = trainGenerator.allDisFeatures;

			List<String> allFeats = gsd.formatAllFeatures(trainFeatureInfo);
			PrintWriter pw = new PrintWriter(new File(dataPath+"svm_train"));

			for (String s : allFeats){
				pw.println(s);
			}
			pw.close();
			
//			pw = new PrintWriter(new File(dataPath+"svm_train_disc_features"));
//
//			for (String s : trainFeatureInfo.allDisFeatures){
//				pw.println(s);
//			}
//			pw.close();

			System.out.println("Generating dev data");
			CorefProperties.setInput(props, Dataset.DEV);
			gsd = new GenerateSVMData(props);
			CoNLLDataGenerator devGenerator = gsd.new CoNLLDataGenerator(gsd.m_splitTrain, true);
			devGenerator.run(props, dictionaries);
			FeatureStruct devFeatureInfo = gsd.new FeatureStruct();
			devFeatureInfo.extractedPositiveFeatures = devGenerator.extractedPositiveFeatures;
			devFeatureInfo.extractedNegativeFeatures = devGenerator.extractedNegativeFeatures;


			devFeatureInfo.allDisFeatures = null;
			devFeatureInfo.allDisFeatures = trainFeatureInfo.allDisFeatures;

			allFeats = gsd.formatAllFeatures(devFeatureInfo);
			pw = new PrintWriter(new File(dataPath+"svm_dev"));

			for (String s : allFeats){
				pw.println(s);
			}
			pw.close();

			pw = new PrintWriter(new File(dataPath+"svm_dev_ids"));

			for (String s : devGenerator.allMentionIds){
				pw.println(s);
			}
			pw.close();
			
			System.out.println("Generating test data");
			CorefProperties.setInput(props, Dataset.TEST);
			gsd = new GenerateSVMData(props);
			CoNLLDataGenerator testGenerator = gsd.new CoNLLDataGenerator(gsd.m_splitTrain, true);
			testGenerator.run(props, dictionaries);
			FeatureStruct testFeatureInfo = gsd.new FeatureStruct();
			testFeatureInfo.extractedPositiveFeatures = testGenerator.extractedPositiveFeatures;
			testFeatureInfo.extractedNegativeFeatures = testGenerator.extractedNegativeFeatures;


			testFeatureInfo.allDisFeatures = null;
			testFeatureInfo.allDisFeatures = trainFeatureInfo.allDisFeatures;

			allFeats = gsd.formatAllFeatures(testFeatureInfo);
			pw = new PrintWriter(new File(dataPath+"svm_test"));

			for (String s : allFeats){
				pw.println(s);
			}
			pw.close();

			pw = new PrintWriter(new File(dataPath+"svm_test_ids"));

			for (String s : testGenerator.allMentionIds){
				pw.println(s);
			}
			pw.close();
			if (gsd.m_wikiCoref){
				props = new Properties();
				props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,mention,coref");
				props.setProperty("tokenize.whitespace", "true");
				props.setProperty("ssplit.eolonly", "true");


				//		    props.setProperty("ner.useSUTime", "false");
				StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
				FeatureStruct wikiFeatureInfo= gsd.new FeatureStruct();
				File docFolder = new File("/data/nlp/moosavne/corpora/WikiCoref/Documents/");
				File[] listOfFiles = docFolder.listFiles();

				pw = new PrintWriter(new File(dataPath+"svm_wiki_test_ids_1"));

				for (int i = 0; i < listOfFiles.length; i++) {
					Samples sample = gsd.parseWikiDocument(pipeline, "/data/nlp/moosavne/corpora/WikiCoref/Annotation/"+listOfFiles[i].getName().replaceAll("\\s", "_")+"/"+listOfFiles[i].getName()+".txt", 
							"/data/nlp/moosavne/corpora/WikiCoref/Annotation/"+listOfFiles[i].getName().replaceAll("\\s", "_")+"/Markables/"+listOfFiles[i].getName()+"_coref_level_OntoNotesScheme.xml");
					wikiFeatureInfo.extractedPositiveFeatures.add(sample.positive);
					wikiFeatureInfo.extractedNegativeFeatures.add(sample.negative);
					for (String s : sample.positiveIds)
						pw.println(s);
					for (String s : sample.negativeIds)
						pw.println(s);

					//				if (i == listOfFiles.length-1){
					//					System.out.println(Arrays.toString(sample.positive.toArray()));
					//					System.out.println(Arrays.toString(sample.negative.toArray()));
					//				}
				}
				pw.close();

				wikiFeatureInfo.allDisFeatures = trainFeatureInfo.allDisFeatures;
				allFeats = gsd.formatAllFeatures(wikiFeatureInfo);
				int positiveSamples = 0, negativeSamples = 0;
				pw = new PrintWriter(new File(dataPath+"svm_wiki_test_1"));
				for (String s : allFeats){
					if (s.startsWith("+1"))
						positiveSamples++;
					else if (s.startsWith("-1"))
						negativeSamples++;
					else
						System.out.println("!!! " + s);
					pw.println(s);
				}
				pw.close();
				System.out.println("Negative samples " + negativeSamples + " positive samples: " + positiveSamples);
			}

		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public Samples parseWikiDocument(StanfordCoreNLP pipeline, String textFile,String markableXMLFile){
		Map<Integer, List<Integer>> docAnaphoricMentionSpans = new HashMap<>();
		Samples samples = new Samples();

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
					wordIndex++;
					sentIndex++;
				}
				sentNum++;
			}
			System.out.println(sysMenNum + " System mention detected");
			int gold_mentions = 0;
			int detected_gold_mentions = 0;
			Set<String> metCorefClusters = new HashSet<>();
			List<Mention> anaphoricMentions = new ArrayList<>();

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
								if (metCorefClusters.contains(corefID)){
									anaphoricMentions.add(indexToMention.get(index));
								}

							}
							//							else
							//								System.out.println(spanStr + " not detected");
						}
						metCorefClusters.add(corefID);

					}
				}
			}
			System.out.println("Gold mentions: " + gold_mentions + " detected gold mentions: " + detected_gold_mentions);
			String docId = textFile.substring(textFile.lastIndexOf("/")+1);

			for (int i = 0; i <allSysMentions.size(); i++){
				Mention m = allSysMentions.get(i);

				boolean positive = anaphoricMentions.contains(m);

				List<String> mFeatures = extractMentionFeatures(allSysMentions, m);
				if (positive){
					samples.positive.add(mFeatures);
					samples.positiveIds.add(docId + " QQQQQQQQQQ "+ m.mentionID);
				}
				else {
					samples.negative.add(mFeatures);
					samples.negativeIds.add(docId + " QQQQQQQQQQ "+ m.mentionID);

				}
			}

		}catch (Exception err) {
			System.out.println(err.getMessage());

		}

		return samples;
	}

	protected List<String> formatDocFeatures(List<List<String>> positive, List<List<String>> negative, List<String> allFeatures){
		List<String> svmData = new ArrayList<String>();

		for (List<String> pos : positive){
			String fVector = "+1 ";
			for (int i = 0; i < allFeatures.size(); i++)
				if (pos.contains(allFeatures.get(i)))
					fVector += (i+1)+":1 ";
			svmData.add(fVector);
		}

		for (List<String> neg : negative){
			String fVector = "-1 ";
			for (int i = 0; i < allFeatures.size(); i++)
				if (neg.contains(allFeatures.get(i)))
					fVector += (i+1)+":1 ";
			svmData.add(fVector);
		}

		return svmData;

	}
	protected List<String> formatAllFeatures(FeatureStruct featStruct) {
		List<String> allFeatVectors = new ArrayList<String>();

		for (int i = 0; i < featStruct.extractedPositiveFeatures.size(); i++){
			allFeatVectors.addAll(formatDocFeatures(featStruct.extractedPositiveFeatures.get(i), featStruct.extractedNegativeFeatures.get(i), featStruct.allDisFeatures));
		}

		return allFeatVectors;
	}

	private class Samples {
		List<List<String>> positive;
		List<String> positiveIds;
		List<List<String>> negative;
		List<String> negativeIds;
		public Samples() {
			positive = new ArrayList<List<String>>();
			positiveIds = new ArrayList<String>();
			negative = new ArrayList<List<String>>();
			negativeIds = new ArrayList<String>();
		}
	}

	private class FeatureStruct {
		List<String> allDisFeatures;
		List<List<List<String>>> extractedPositiveFeatures;
		List<List<List<String>>> extractedNegativeFeatures;

		public FeatureStruct(){
			allDisFeatures = new ArrayList<String>();
			extractedPositiveFeatures = new ArrayList<List<List<String>>>();
			extractedNegativeFeatures = new ArrayList<List<List<String>>>();
		}
	}

	class CoNLLDataGenerator implements CorefDocumentProcessor{
		List<String> allDisFeatures = new ArrayList<String>();
		List<List<List<String>>> extractedPositiveFeatures = new ArrayList<List<List<String>>>();
		List<List<List<String>>> extractedNegativeFeatures = new ArrayList<List<List<String>>>();
		List<String> allMentionIds = new ArrayList<>();
		int positiveSampleSize = 0, negSampleSize=0;
		Map<String, Integer> discPosFeatures = new HashMap<String, Integer>();
		Map<String, Integer> discNegFeatures = new HashMap<String, Integer>();
		int prunThreshold = 10;
		boolean m_split;
		boolean m_even;



		public CoNLLDataGenerator(boolean split, boolean even) {
			m_split = split;
			m_even = even;
		}

		@Override
		public void process(int id, Document document) {
			if (m_split && ((m_even && id%2==1)|| (!m_even && id%2==0))){
				return;
			}
			Samples sample = generateDocFeatures(document);
			extractedPositiveFeatures.add(sample.positive);
			positiveSampleSize += sample.positive.size();
			extractedNegativeFeatures.add(sample.negative);
			negSampleSize += sample.negative.size();

			allMentionIds.addAll(sample.positiveIds);
			allMentionIds.addAll(sample.negativeIds);

			for (List<String> pos : sample.positive){
				for (String f : pos){
					int count = 1;
					if (discPosFeatures.containsKey(f)){
						count = discPosFeatures.get(f)+1;
					}
					discPosFeatures.put(f, count);
				}
			}

			for (List<String> neg : sample.negative){
				for (String f : neg){
					int count = 1;
					if (discNegFeatures.containsKey(f))
						count = discNegFeatures.get(f)+1;
					discNegFeatures.put(f, count);
				}
			}

		}

		@Override
		public void finish() throws Exception {
			// TODO Auto-generated method stub

		}


		Samples generateDocFeatures(Document document) {

			Samples samples = new Samples();
			document.extractGoldCorefClusters();

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

			for (Mention predictedMention : mentionsList){
				boolean positive = false;

				if ( allCoreferentMentions.contains(predictedMention.mentionID) && (!m_anaphoricity || !allFirstMentions.contains(predictedMention.mentionID)))
					positive = true;

				List<String> mFeatures = extractMentionFeatures(mentionsList, predictedMention);
				if (positive){
					samples.positive.add(mFeatures);
					samples.positiveIds.add(document.conllDoc.documentID +document.conllDoc.getPartNo()+ " QQQQQQQQQQ "+predictedMention.mentionID );
				}
				else{
					samples.negative.add(mFeatures);
					samples.negativeIds.add(document.conllDoc.documentID +document.conllDoc.getPartNo()+ " QQQQQQQQQQ "+predictedMention.mentionID);
				}
				//					String featVector = "";
				//
				//					if (positive)
				//						featVector = "+1 ";
				//					else
				//						featVector = "-1 ";
				//
				//					for (int f = 0; f < mFeatures.size(); f++)
				//						featVector += f +":"+mFeatures.get(f) + " ";
				//
				//					feats.add(featVector);			
			}

			return samples;
		}

	}

	private final String sentenceStartWord = "BEG";
	private final String sentenceEndWord = "END";

	public List<String> extractMentionFeatures(Collection<Mention> allMentions, Mention m){
		List<String> features = new ArrayList<String>();

		if (flags.useWholeNPText)
			features.add((flags.useLowerCasedStrings ? m.spanToString().toLowerCase() : m.spanToString())+"-WholeNPText");

		if (flags.useText){

			for (int i = m.startIndex; i < m.endIndex-1; i++){
				String s = m.sentenceWords.get(i).get(CoreAnnotations.TextAnnotation.class);
				features.add((flags.useLowerCasedStrings ? s.toLowerCase() : s)+"-MentionWord"+(i-m.startIndex));
			}

			String lw = m.sentenceWords.get(m.endIndex-1).get(CoreAnnotations.TextAnnotation.class);
			features.add((flags.useLowerCasedStrings ? lw.toLowerCase() : lw)+"-MentionLastWord");

			if (flags.useHead)
				features.add((flags.useLowerCasedStrings ? m.headString.toLowerCase() : m.headString)+"-HeadText");

			String prefix = "";
			for (int i = 1; i <= flags.windowSize; i++){
				String s = "";
				if (m.startIndex-i >= 0)
					s = m.sentenceWords.get(m.startIndex-i).get(CoreAnnotations.TextAnnotation.class);
				else
					s = sentenceStartWord;
				prefix +="P";
				features.add((flags.useLowerCasedStrings ? s.toLowerCase() : s) + "-" + prefix +"WordText");
			}

			prefix = "";
			for (int i = 0; i < flags.windowSize; i++){
				String s = "";
				if (m.endIndex+i <m.sentenceWords.size())
					s = m.sentenceWords.get(m.endIndex+i).get(CoreAnnotations.TextAnnotation.class);
				else
					s = sentenceEndWord;
				prefix +="N";
				features.add((flags.useLowerCasedStrings ? s.toLowerCase() : s) + "-" + prefix +"WordText");
			}
		}

		if (flags.useLemma){
			if (flags.useWholeNPLemma){
				String NPLemma = "";

				for (int i = 0; i < m.originalSpan.size(); i++){
					String lemma = m.originalSpan.get(i).get(CoreAnnotations.LemmaAnnotation.class); 
					NPLemma += lemma  + " ";
				}
				features.add(NPLemma.trim()+"-WholeNPLemma");
			}

			if (flags.useHead){
				String headLemma = m.headIndex >= 0 ? m.sentenceWords.get(m.headIndex).get(CoreAnnotations.LemmaAnnotation.class) : "null";
				features.add(headLemma +"-HeadLemma");
			}

			if (flags.useNewInnerWordRepresentation){

				String ll = m.sentenceWords.get(m.endIndex-1).get(CoreAnnotations.LemmaAnnotation.class);
				features.add(ll+"-MentionLastLemma");
				String fl = m.sentenceWords.get(m.startIndex).get(CoreAnnotations.LemmaAnnotation.class);
				features.add(fl+"-MentionFirstLemma");


				for (int i = m.startIndex+1; i < m.headIndex; i++){
					String s = m.sentenceWords.get(i).get(CoreAnnotations.LemmaAnnotation.class);
					features.add(s+"-PreHeadLemma");
				}

				for (int i = m.headIndex+1; i < m.endIndex-1; i++){
					String s = m.sentenceWords.get(i).get(CoreAnnotations.LemmaAnnotation.class);
					features.add(s+"-PostHeadLemma");
				}

			}
			else {
				for (int i = m.startIndex; i < m.endIndex-1; i++){
					String s = m.sentenceWords.get(i).get(CoreAnnotations.LemmaAnnotation.class);
					features.add(s+"-MentionLemma"+(i-m.startIndex));
				}
				String lw = m.sentenceWords.get(m.endIndex-1).get(CoreAnnotations.LemmaAnnotation.class);
				features.add(lw+"-MentionLastLemma");

			}

			String prefix = "";
			for (int i = 1; i <= flags.windowSize; i++){
				String s = "";
				if (m.startIndex-i >= 0)
					s = m.sentenceWords.get(m.startIndex-i).get(CoreAnnotations.LemmaAnnotation.class);
				else
					s = sentenceStartWord;
				prefix +="P";
				features.add(s + "-" + prefix +"WordLemma");
			}

			prefix = "";
			for (int i = 0; i < flags.windowSize; i++){
				String s = "";
				if (m.endIndex+i <m.sentenceWords.size())
					s = m.sentenceWords.get(m.endIndex+i).get(CoreAnnotations.LemmaAnnotation.class);
				else
					s = sentenceEndWord;
				prefix +="N";
				features.add(s + "-" + prefix +"WordLemma");
			}

		}

		if (flags.useTags){

			if (flags.useWholeNPTags){
				String NPTags = "";

				for (int i = 0; i < m.originalSpan.size(); i++){
					String lemma = m.originalSpan.get(i).get(CoreAnnotations.PartOfSpeechAnnotation.class); 
					NPTags += lemma  + " ";
				}
				features.add(NPTags.trim()+"-WholeNPTags");
			}

			if (flags.useHead){
				String headTag = m.headIndex >= 0 ? m.sentenceWords.get(m.headIndex).get(CoreAnnotations.PartOfSpeechAnnotation.class) : "null";
				features.add(headTag +"-HeadTag");
			}

			if (flags.useNewInnerWordRepresentation){

				String ll = m.sentenceWords.get(m.endIndex-1).get(CoreAnnotations.PartOfSpeechAnnotation.class);
				features.add(ll+"-MentionLastTag");
				String fl = m.sentenceWords.get(m.startIndex).get(CoreAnnotations.PartOfSpeechAnnotation.class);
				features.add(fl+"-MentionFirstTag");


				for (int i = m.startIndex+1; i < m.headIndex; i++){
					String s = m.sentenceWords.get(i).get(CoreAnnotations.PartOfSpeechAnnotation.class);
					features.add(s+"-PreHeadWordTag");
				}

				for (int i = m.headIndex+1; i < m.endIndex-1; i++){
					String s = m.sentenceWords.get(i).get(CoreAnnotations.PartOfSpeechAnnotation.class);
					features.add(s+"-PostHeadWordTag");
				}

			}
			else {
				for (int i = m.startIndex; i < m.endIndex-1; i++){
					String s = m.sentenceWords.get(i).get(CoreAnnotations.PartOfSpeechAnnotation.class);
					features.add(s+"-MentionTags"+(i-m.startIndex));
				}
				String lw = m.sentenceWords.get(m.endIndex-1).get(CoreAnnotations.PartOfSpeechAnnotation.class);
				features.add(lw+"-MentionLastTags");

			}



			String prefix = "";
			for (int i = 1; i <= flags.windowSize; i++){
				String s = "";
				if (m.startIndex-i >= 0)
					s = m.sentenceWords.get(m.startIndex-i).get(CoreAnnotations.PartOfSpeechAnnotation.class);
				else
					s = sentenceStartWord;
				prefix +="P";
				features.add(s + "-" + prefix +"WordTag");
			}

			prefix = "";
			for (int i = 0; i < flags.windowSize; i++){
				String s = "";
				if (m.endIndex+i <m.sentenceWords.size())
					s = m.sentenceWords.get(m.endIndex+i).get(CoreAnnotations.PartOfSpeechAnnotation.class);
				else
					s = sentenceEndWord;
				prefix +="N";
				features.add(s + "-" + prefix +"WordTag");
			}
		}

		if (flags.useVerbs){
			for (int i = m.startIndex-1; i >=0 ; i--){
				if (m.sentenceWords.get(i).get(CoreAnnotations.PartOfSpeechAnnotation.class).toLowerCase().startsWith("v")){
					features.add(m.sentenceWords.get(i).get(CoreAnnotations.LemmaAnnotation.class)+"-PVerb");
					break;
				}
			}
			for (int i = m.endIndex; i < m.sentenceWords.size(); i++){
				if (m.sentenceWords.get(i).get(CoreAnnotations.PartOfSpeechAnnotation.class).toLowerCase().startsWith("v")){
					features.add(m.sentenceWords.get(i).get(CoreAnnotations.LemmaAnnotation.class)+"-NVerb");
					break;
				}
			}

		}

		if (flags.useNERTags){
			features.add(m.nerString+"-NERTag");
		}

		if (flags.useLength){
			features.add(((flags.maxMentionLength!= -1 && m.originalSpan.size() > flags.maxMentionLength) ? flags.maxMentionLength : m.originalSpan.size())+"-MentionLength");
		}

		if (flags.usePosition)
			features.add(m.getPosition()+"-SentencePosition");

		//		if (flags.useLinguistic){
		//			List<String> linguisticFeatures = m.getSingletonFeatures(dictionaries);
		//			for (int i = 0; i< linguisticFeatures.size(); i++)
		//				features.add(linguisticFeatures.get(i)+"-Linguistic"+i);
		//		}

		if (flags.useNGrams){
			Collection<String> subs = null;
			String cWord = m.spanToString();
			//			if (flags.cacheNGrams) {
			//				subs = wordToSubstrings.get(cWord);
			//			}
			if (subs == null) {
				subs = new ArrayList<String>();
				String word = '<' + cWord + '>';
				if (flags.lowercaseNGrams) {
					word = word.toLowerCase();
				}
				if (flags.dehyphenateNGrams) {
					word = dehyphenate(word);
				}
				if (flags.greekifyNGrams) {
					word = greekify(word);
				}
				// minimum length substring is 2 letters (hardwired)
				// hoist flags.noMidNGrams so only linear in word length for that case
				if (flags.noMidNGrams) {
					int max = flags.maxNGramLeng >= 0 ? Math.min(flags.maxNGramLeng, word.length()) :
						word.length();
					for (int j = 2; j <= max; j++) {
						subs.add(intern('#' + word.substring(0, j) + '#'));
					}
					int start = flags.maxNGramLeng >= 0 ? Math.max(0, word.length() - flags.maxNGramLeng) :
						0;
					int lenM1 = word.length() - 1;
					for (int i = start; i < lenM1; i++) {
						subs.add(intern('#' + word.substring(i) + '#'));
					}
				} else {
					for (int i = 0; i < word.length(); i++) {
						for (int j = i + 2, max = Math.min(word.length(), i + flags.maxNGramLeng); j <= max; j++) {
							if (flags.maxNGramLeng >= 0 && j - i > flags.maxNGramLeng) {
								continue;
							}
							subs.add(intern('#' + word.substring(i, j) + '#'));
						}
					}
				}
				//				if (flags.cacheNGrams) {
				//					wordToSubstrings.put(cWord, subs);
				//				}
			}
			features.addAll(subs);

		}

		if (flags.useNPType)
			features.add(m.mentionType.toString()+"-MentionType");

		if (flags.useTextMatch){
			boolean appearedAgain = false;
			boolean appearBefore = false;
			boolean anaInAnte = false;
			boolean anteInAna = false;
			boolean anaInPrevAnte = false, anteInPreAna = false;
			for (Mention ante : allMentions){
				if (!ante.equals(m)&& (ante.sentNum != m.sentNum || (ante.sentNum == m.sentNum && ante.startIndex !=m.startIndex && ante.endIndex != m.endIndex))){
					boolean earlier = ante.appearEarlierThan(m);
						if (ante.spanToString().equalsIgnoreCase(m.spanToString())){
						appearedAgain = true;
						if (earlier)
							appearBefore = true;
						}
						if (m.spanToString().toLowerCase().contains(ante.spanToString().toLowerCase())){
							anteInAna = true;
							if (earlier)
								anteInPreAna  = true;

						}
						if (ante.spanToString().toLowerCase().contains(m.spanToString().toLowerCase())){
							anaInAnte = true;
							if (earlier)
								anaInPrevAnte  = true;
						}
					}
			}
			features.add(appearedAgain ? "MentionAppearedAgain" : "MentionNotAppeared");
			features.add(appearBefore ? "MentionAppearedBefore" : "MentionNotAppearedBefore");
			features.add(anaInAnte ? "MentionisContained" : "MentionisNOTContained");
			features.add(anaInPrevAnte ? "MentionisContainedBefore" : "MentionisNOTContainedBefore");
			features.add(anteInAna ? "MentionContains" : "MentionNOTContains");
			features.add(anteInPreAna ? "MentionContainsBefore" : "MentionNOTContainsBefore");

		}

		if (flags.useHeadMatch){
			boolean headAppeared = false;
			boolean headAppearedBefore = false;
			for (Mention men : allMentions){
				if (!men.equals(m))
					if ((men.sentNum != m.sentNum || (men.sentNum == m.sentNum && men.startIndex !=m.startIndex && men.endIndex != m.endIndex))
							&& men.headString.equalsIgnoreCase(m.headString)){
						headAppeared = true;
						if (men.appearEarlierThan(m))
							headAppearedBefore = true;
						break;
					}
			}
			features.add(headAppeared ? "HeadAppearedAgain" : "HeadNotAppeared");
			features.add(headAppearedBefore ? "HeadAppearedBefore" : "HeadNotAppearedBefore");
		}

		return features;
	}

	String intern(String s) {
		if (flags.intern) {
			return s.intern();
		} else {
			return s;
		}
	}

	private String dehyphenate(String str) {
		// don't take out leading or ending ones, just internal
		// and remember padded with < > characters
		String retStr = str;
		int leng = str.length();
		int hyphen = 2;
		do {
			hyphen = retStr.indexOf('-', hyphen);
			if (hyphen >= 0 && hyphen < leng - 2) {
				retStr = retStr.substring(0, hyphen) + retStr.substring(hyphen + 1);
			} else {
				hyphen = -1;
			}
		} while (hyphen >= 0);
		return retStr;
	}

	private String greekify(String str) {
		// don't take out leading or ending ones, just internal
		// and remember padded with < > characters

		String pattern = "(alpha)|(beta)|(gamma)|(delta)|(epsilon)|(zeta)|(kappa)|(lambda)|(rho)|(sigma)|(tau)|(upsilon)|(omega)";

		Pattern p = Pattern.compile(pattern);
		Matcher m = p.matcher(str);
		return m.replaceAll("~");
	}

}


