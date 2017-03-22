package wikiCoref;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Pattern;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.*;

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.coref.data.Dictionaries.MentionType;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

public class NNDataGenerator {
	private PrintWriter dataWriter;
	StanfordCoreNLP pipeline;

	public NNDataGenerator(String dataPath){
		try {
			dataWriter = new PrintWriter(dataPath);
			Properties props = new Properties();
			props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,mention,coref");
			props.setProperty("tokenize.whitespace", "true");
			props.setProperty("ssplit.eolonly", "true");
			props.setProperty("coref.algorithm", "neural");
			//		    props.setProperty("ner.useSUTime", "false");
			pipeline = new StanfordCoreNLP(props);

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void parseDocument(String textFile,String markableXMLFile){
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

			for (int i = 0; i <allSysMentions.size(); i++){
				Mention m = allSysMentions.get(i);
				dataWriter.println(getMentionRepresentation(docId,m, sentences, allSysMentions.subList(0, i), anaphoricMentions.contains(m)));
			}
//			pw.close();
		}catch (Exception err) {
			System.out.println(err.getMessage());

		}

		return; 
	}

	private String getMentionRepresentation(String docId, Mention m, List<CoreMap> sentences, List<Mention> antecedents, boolean isAnaphoric){
		String separator = " QQQQQQQQQQ ";

		boolean[] featVals =  anaphoricityPairwiseFeatures(m, antecedents);

		String line = docId + separator + m.mentionID + separator + contextRepresentation(m, sentences)+separator;
		line += m.spanToString().toLowerCase() + separator;
		line += m.headString.toLowerCase() ;

		for (int i = 0; i < featVals.length; i++){
			if (featVals[i])
				line+= separator + "1";
			else
				line+= separator + "0";
		}
		Iterator<SemanticGraphEdge> iterator =
	              m.enhancedDependency.incomingEdgeIterator(m.headIndexedWord);
	          SemanticGraphEdge relation = iterator.hasNext() ? iterator.next() : null;

		line += separator + m.mentionType;
		line  += separator + m.mentionNum;
		line += separator + "0";
		line += separator + fine_type(m);
		line += separator + "null";
		line += separator + (relation == null ? "<missing>" : relation.getRelation().getShortName());

		int label = 0;
		if (isAnaphoric)
			label = 1;

		line += separator + label;	

		return line;
	}

	public boolean[] anaphoricityPairwiseFeatures(Mention m1, List<Mention> antecedents) {
		boolean[] features = new boolean[11];

		for (int i = 0; i < features.length; i++)
			features[i] = false;

		for (Mention ante : antecedents){
			boolean earlier = ante.appearEarlierThan(m1);
			if (ante != m1 && m1.insideIn(ante))
				features[10] = true;
			
			if (ante != m1 && !(m1.insideIn(ante) || ante.insideIn(m1))){
				if (m1.toString().trim().equalsIgnoreCase(ante.toString().trim())){
					features[0] = true;
					if (earlier)
						features[1] = true;
				}

				if (m1.headString.equalsIgnoreCase(ante.headString)){
					features[2] = true;
					if (earlier)
						features[3] = true;
				}

				//				if (m1.headsAgree(ante)){
				//					features[4] = true;
				//					if (earlier)
				//						features[5] = true;
				//				}

				if (m1.headString.toLowerCase().contains(ante.headString.toLowerCase()) || ante.headString.toLowerCase().contains(m1.headString.toLowerCase())){
					features[4] = true;
					if (earlier)
						features[5] = true;
				}

				if (m1.spanToString().toLowerCase().contains(ante.spanToString().toLowerCase())){
					features[6] = true;
					if (earlier)
						features[7]  = true;

				}
				if (ante.spanToString().toLowerCase().contains(m1.spanToString().toLowerCase())){
					features[8] = true;
					if (earlier)
						features[9]  = true;

				}
				//				if(edu.stanford.nlp.coref.statistical.FeatureExtractor.relaxedStringMatch(m1, ante)){
				//					features[10] = true;
				//					if(earlier)
				//						features[11] = true;
				//				}
			}


		}

		return features;
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
	
	public String contextRepresentation(Mention m, List<CoreMap> sentences){
		int prevContextSize = 100;
		int follContextSize = 10;

		List<String> prevWords = new ArrayList<String>();
		List<String> follWords = new ArrayList<String>();

		int startIdx = m.startIndex < prevContextSize ? 0 : m.startIndex - prevContextSize;
		int endIdx = m.sentenceWords.size() - m.endIndex < follContextSize ? m.sentenceWords.size() : m.endIndex+follContextSize;

		for (int i = m.startIndex; i >= startIdx; i--)
			prevWords.add(m.sentenceWords.get(i).get(CoreAnnotations.TextAnnotation.class).toLowerCase());
		for (int i = m.endIndex; i < endIdx; i++)
			follWords.add(m.sentenceWords.get(i).get(CoreAnnotations.TextAnnotation.class).toLowerCase());

		if (prevWords.size() < prevContextSize){
			int cnt = prevContextSize - prevWords.size();
			for (int i = m.sentNum-1; i > 0 && cnt > 0; i--){
				List<CoreLabel> words = sentences.get(i).get(CoreAnnotations.TokensAnnotation.class);
				for (int j = words.size()-1; j >= 0 && cnt > 0; j--){
					prevWords.add(words.get(j).get(CoreAnnotations.TextAnnotation.class).toLowerCase());
					cnt--;
				}
			}        
		}

		if (follWords.size() < follContextSize){
			int cnt = follContextSize - follWords.size();
			for (int i = m.sentNum+1; i < sentences.size() && cnt > 0; i++){
				List<CoreLabel> words = sentences.get(i).get(CoreAnnotations.TokensAnnotation.class);
				for (int j = 0; j < words.size() && cnt > 0; j++){
					follWords.add(words.get(j).get(CoreAnnotations.TextAnnotation.class).toLowerCase());
					cnt--;
				}
			}
		}
		String context = "";
		for (int i = prevWords.size()-1; i> 0; i--)
			context += prevWords.get(i) + " ";
		context += " *MENTIONHERE* ";

		for (int i = 0; i < follWords.size(); i++)
			context += follWords.get(i) + " ";

		return context;
	}

	public static void main(String[] args){
		NNDataGenerator nn = new NNDataGenerator(args[0]);
		File docFolder = new File("/data/nlp/moosavne/corpora/WikiCoref/Documents/");
		File[] listOfFiles = docFolder.listFiles();

		for (int i = 0; i < listOfFiles.length; i++) {
			nn.parseDocument("/data/nlp/moosavne/corpora/WikiCoref/Annotation/"+listOfFiles[i].getName().replaceAll("\\s", "_")+"/"+listOfFiles[i].getName()+".txt", 
					"/data/nlp/moosavne/corpora/WikiCoref/Annotation/"+listOfFiles[i].getName().replaceAll("\\s", "_")+"/Markables/"+listOfFiles[i].getName()+"_coref_level_OntoNotesScheme.xml");
		}
//		nn.parseDocument(args[1], args[2]);
		nn.dataWriter.close();

	}
}
