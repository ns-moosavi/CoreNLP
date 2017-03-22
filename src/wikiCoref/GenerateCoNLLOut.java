package wikiCoref;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
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

import com.sun.net.httpserver.Filter.Chain;

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.CorefCoreAnnotations.CorefChainAnnotation;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.coref.data.CorefChain.CorefMention;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

public class GenerateCoNLLOut {
	private PrintWriter dataWriter;
	StanfordCoreNLP pipeline;
	public GenerateCoNLLOut(String dataPath){
		try {
			dataWriter = new PrintWriter(dataPath+"_deep_coref.out");
			Properties props = new Properties();
			props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,mention,coref");
			props.setProperty("tokenize.whitespace", "true");
			props.setProperty("ssplit.eolonly", "true");
			props.setProperty("coref.algorithm", "neural");
			props.setProperty("coref.neural.modelPath", "/data/nlp/moosavne/git/CoreNLP/exported_weights/model.ser.gz");
			props.setProperty("coref.neural.embeddingsPath", "/data/nlp/moosavne/git/CoreNLP/exported_weights/embeddings.ser.gz");

			//		    props.setProperty("ner.useSUTime", "false");
			pipeline = new StanfordCoreNLP(props);

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void parseDocument(String textFile){

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
			System.out.println("Annotated");
			Map<Integer, CorefChain> corefChains = document.get(CorefChainAnnotation.class);
			dataWriter.println("#begin document "+docId);
			Map<Integer, Map<Integer, String>> mentionIndexToChainId = new HashMap<Integer, Map<Integer,String>>();
			for (Integer chainId: corefChains.keySet()){
				for (CorefMention men : corefChains.get(chainId).getMentionsInTextualOrder()){
					if(!mentionIndexToChainId.containsKey(men.sentNum))
						mentionIndexToChainId.put(men.sentNum, new HashMap<>());
					if (men.endIndex-men.startIndex == 1){
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
			List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

			for (int j = 0; j < sentences.size();j++){
				if (mentionIndexToChainId.containsKey(j+1)){
					for (int k = 0; k < sentences.get(j).get(CoreAnnotations.TokensAnnotation.class).size();k++)
						if (mentionIndexToChainId.get(j+1).containsKey(k+1))
							dataWriter.println("Nan\t0\t"+(k+1)+"\t"+ sentences.get(j).get(CoreAnnotations.TokensAnnotation.class).get(k).originalText()+"\t"+mentionIndexToChainId.get(j+1).get(k+1));
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

		}catch (Exception err) {
			System.out.println(err.getMessage());

		}

		return; 
	}
	public static void main(String[] args){
		GenerateCoNLLOut nn = new GenerateCoNLLOut(args[0]);
		File docFolder = new File("/data/nlp/moosavne/corpora/WikiCoref/Documents/");
		File[] listOfFiles = docFolder.listFiles();

		for (int i = 0; i < listOfFiles.length; i++) {
			System.out.println(listOfFiles[i].getName().replaceAll("\\s", "_")+"/"+listOfFiles[i].getName());
			nn.parseDocument("/data/nlp/moosavne/corpora/WikiCoref/Annotation/"+listOfFiles[i].getName().replaceAll("\\s", "_")+"/"+listOfFiles[i].getName()+".txt");
		}
		//		nn.parseDocument(args[1], args[2]);
				nn.dataWriter.close();

	}
}
