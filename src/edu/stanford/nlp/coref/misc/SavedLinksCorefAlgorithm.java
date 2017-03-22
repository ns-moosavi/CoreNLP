package edu.stanford.nlp.coref.misc;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import edu.stanford.nlp.coref.CorefAlgorithm;
import edu.stanford.nlp.coref.CorefSystem;
import edu.stanford.nlp.coref.CorefUtils;
import edu.stanford.nlp.coref.data.Document;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;

public class SavedLinksCorefAlgorithm implements CorefAlgorithm {
  private final Map<Integer, List<Pair<Integer, Integer>>> toMerge = new HashMap<>();
  private int currentDocId = 0;
  private PrintWriter pw;
  private PrintWriter corr_pw;

  public SavedLinksCorefAlgorithm(String savedLinkPath, PrintWriter pw, PrintWriter correctDecisionWriter) {
	  this.pw = pw;
	  this.corr_pw = correctDecisionWriter;
	  
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

  @Override
  public void runCoref(Document document) {
    if (toMerge.containsKey(currentDocId)) {
  	  List<CoreMap> sentences = document.annotation.get(SentencesAnnotation.class);
  	  System.out.println(currentDocId);
      for (Pair<Integer, Integer> pair : toMerge.get(currentDocId)) {
        CorefUtils.mergeCoreferenceClusters(pair, document);
        if (pw != null){
        	Mention m1 = document.predictedMentionsByID.get(pair.first);
        	Mention m2 = document.predictedMentionsByID.get(pair.second);
        	int m1Start = 0, m2Start = 0;
        	for (int k = 0; k < sentences.size();k++){
        		int siz = sentences.get(k).get(CoreAnnotations.TokensAnnotation.class).size();
        		if (k < m1.sentNum){
        			m1Start+= siz;
        		}
        		if (k < m2.sentNum)
        			m2Start += siz;
        		
        	}
        	m1Start += m1.startIndex;
        	m2Start += m2.startIndex;
        	
        	pw.write("("+document.conllDoc.documentID +"); part "+String.format("%03d", Integer.parseInt(document.conllDoc.getPartNo()))+  "\t" +
                    "("+m2Start+", "+ (m2Start+m2.originalSpan.size()-1) +")"+ "\t" +
        			"("+m1Start+", "+(m1Start+m1.originalSpan.size()-1)+")" + "\n");
        }
        if (corr_pw != null){
        	Mention m1 = document.predictedMentionsByID.get(pair.first);
        	Mention m2 = document.predictedMentionsByID.get(pair.second);
                boolean are_coref = false;
		if(document.goldMentionsByID.containsKey(m1.mentionID) && document.goldMentionsByID.containsKey(m2.mentionID) &&
			document.goldMentionsByID.get(m1.mentionID).goldCorefClusterID == document.goldMentionsByID.get(m2.mentionID).goldCorefClusterID)
			are_coref = true;
        	if (!m1.headString.equalsIgnoreCase(m2.headString))
        		corr_pw.println((are_coref ? 1 : 0) + "\t"+m1.mentionType + "\t" + m2.mentionType+"\t"+m1.spanToString() + " " + m2.spanToString() + "\t" + m1.headString + " " + m2.headString + "\t"+ getContext(m1) + " " + getContext(m2));
        }
      }
    }
    currentDocId += 1;
  }
  
  private String getContext (Mention m1){
	  String m1Context = 
			  //(m1.startIndex-2 >=0 ? m1.sentenceWords.get(m1.startIndex-2).get(CoreAnnotations.TextAnnotation.class).toLowerCase(): "NONE" )
  			(m1.startIndex-1 >=0 ? m1.sentenceWords.get(m1.startIndex-1).get(CoreAnnotations.TextAnnotation.class).toLowerCase(): "NONE" )
  			+ " MENTION " + 
  			(m1.endIndex < m1.sentenceWords.size() ? m1.sentenceWords.get(m1.endIndex).get(CoreAnnotations.TextAnnotation.class).toLowerCase(): "NONE"); 
  			//+ " " + (m1.endIndex+1 < m1.sentenceWords.size() ? m1.sentenceWords.get(m1.endIndex+1).get(CoreAnnotations.TextAnnotation.class).toLowerCase(): "NONE");
	  
	  return m1Context;
  }
  
  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(new String[] {"-props", args[0]});
    PrintWriter pw = new PrintWriter("deep_coref_test.ante");
    PrintWriter corr_dec = null;//new PrintWriter("deep_coref_non_head_match_made_decisions_on_dev.txt");
    
    CorefSystem coref = new CorefSystem(props, new SavedLinksCorefAlgorithm(args[1], pw, corr_dec));
    coref.runOnConll(props);
   	pw.close();
//    corr_dec.close();
  }
}
