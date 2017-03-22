package edu.stanford.nlp.coref.misc;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.coref.CorefDocumentProcessor;
import edu.stanford.nlp.coref.CorefProperties;
import edu.stanford.nlp.coref.CorefUtils;
import edu.stanford.nlp.coref.CorefProperties.Dataset;
import edu.stanford.nlp.coref.data.Dictionaries;
import edu.stanford.nlp.coref.data.Document;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.coref.data.Dictionaries.MentionType;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;

public class CoNLLAnalysis implements CorefDocumentProcessor{
	Map<String, Set<String>> docMentionMap;
	Map<String, Integer> mentionDocCount;
	int allDoc = 0;
	//	Map<String, Set<String>> docMentionMap;
	boolean usePairs = false;
	String currentDataset = "";

	List<String> beSkipped = Arrays.asList("this", "those", "year", "tomorrow", "today", "person", "that", "these", "one", "today", "number" , 
			"man", "woman", "place", "like", "week", "day", "time", "women", "men", "some", "all", 
			"thing", "sides", "people", "place", "two", "tonight", "yesterday",  "january", "unit", "call", "boy", "both", 
			"saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "february", "march", "april", "may", "june", "july", 
			"august", "september", "november", "october", "december", "child", "mother", "daughter", "brother", "end", "year", "years", "period", "jesus", "god",
			"thousands", "system", "other", "everyone", "times", "any", "each", "another", "thing", "things", "case", "lot", "dad", "minute", "minutes"
			, "part","someone", "1st", "you", "he", "one", "two", "three", "four", "five");
	
	public CoNLLAnalysis() {
		//		docMentionMap = new HashMap<String, Set<String>>();
		docMentionMap = new HashMap<String, Set<String>>();
		mentionDocCount = new HashMap<String, Integer>();

	}

	@Override
	public void process(int id, Document document) {
		allDoc++;
		String sid = document.conllDoc.documentID+"\t"+document.conllDoc.getPartNo()+"\t"+document.goldMentions.size();
		if (usePairs){
			Map<Pair<Integer, Integer>, Boolean> mentionPairs = CorefUtils.getLabeledMentionPairs(document);
			Set<String> headPairSet = new HashSet<String>();
			for (Map.Entry<Pair<Integer, Integer>, Boolean> e : mentionPairs.entrySet()) {
				Mention m1 = document.predictedMentionsByID.get(e.getKey().first);
				Mention m2 = document.predictedMentionsByID.get(e.getKey().second);
				if (e.getValue() && !(m1.mentionType == MentionType.PRONOMINAL || m2.mentionType == MentionType.PRONOMINAL)){
					String hp1 = m1.headString+"\t"+m2.headString;
					String hp2 = m2.headString+"\t"+m1.headString;
					headPairSet.add(hp1);
					headPairSet.add(hp2);
				}
			}
			docMentionMap.put(sid, headPairSet);
		}
		else{
			List<Mention> mentionList = CorefUtils.getSortedMentions(document);
			Set<String> nonPronounMentions = new HashSet<String>();

			for (int i = 0; i < mentionList.size(); i++){
				Mention m1 = mentionList.get(i);

				boolean isCoreferent = document.goldMentionsByID.containsKey(m1.mentionID);
				if (isCoreferent && (m1.mentionType != MentionType.PRONOMINAL)){
					nonPronounMentions.add(m1.headString);
				}
			}
			docMentionMap.put(sid, nonPronounMentions);
			for (String s : nonPronounMentions){
				if (!mentionDocCount.containsKey(s))
					mentionDocCount.put(s, 0);
				mentionDocCount.put(s, mentionDocCount.get(s)+1);
			}
		}

	}

	@Override
	public void finish() throws Exception {
		// TODO Auto-generated method stub

	}

	public static void main(String[] args){
		Properties props = StringUtils.argsToProperties(new String[] {"-props", args[0]});
		Dictionaries dictionaries;
		try {

			dictionaries = new Dictionaries(props);

			CoNLLAnalysis analizer = new CoNLLAnalysis();
			CorefProperties.setInput(props, Dataset.TRAIN);
			analizer.currentDataset = "train";
			analizer.run(props, dictionaries);
			CorefProperties.setInput(props, Dataset.DEV);
			analizer.currentDataset = "dev";
			analizer.run(props, dictionaries);
			CorefProperties.setInput(props, Dataset.TEST);
			analizer.currentDataset = "test";
			analizer.run(props, dictionaries);
			List<String> docList = new ArrayList<String>();
			docList.addAll(analizer.docMentionMap.keySet());
			Map<String, List<String>> similarDocs = new HashMap<String, List<String>>();

			Map<String, Set<String>> toBeMerged = new HashMap<String, Set<String>>();
			Map<String, Integer> docPart = new HashMap<String, Integer>();
			Map<String, Integer> docMentionNum = new HashMap<String, Integer>();
			Set<String> veryFrequent = new HashSet<String>();
			
			
//			double freqThreshold = 50;//0.5 * analizer.allDoc; 
//			for (String s : analizer.mentionDocCount.keySet()){
//				if (analizer.mentionDocCount.get(s) > freqThreshold)
//					veryFrequent.add(s);
//			}
//			System.out.println(Arrays.toString(veryFrequent.toArray()));
			
			for (int i= 0; i < docList.size(); i++){
				String fid = docList.get(i);
				String fDocNam = fid.split("\t")[0];

				docMentionNum.put(fDocNam, Integer.parseInt(fid.split("\t")[2]));
				for (int j = i+1; j < docList.size(); j++){
					String sid = docList.get(j);
					String sDocNam = sid.split("\t")[0];

					if (fDocNam.equals(sDocNam)){
						if (!toBeMerged.containsKey(fDocNam))
							toBeMerged.put(fDocNam, new HashSet<String>());
						toBeMerged.get(fDocNam).add(fid);
						toBeMerged.get(fDocNam).add(sid);
					}
				}
			}

			for (String s : toBeMerged.keySet()){
				String partList="";
				int mentionNum = 0;
				Set<String> uMentions = new HashSet<String>();
				docPart.put(s, toBeMerged.get(s).size());
				for (String d : toBeMerged.get(s)){
					uMentions.addAll(analizer.docMentionMap.get(d));
					mentionNum += Integer.parseInt(d.split("\t")[2]);
					partList += d.split("\t")[1]+"+";
					analizer.docMentionMap.remove(d);
				}
				docMentionNum.put(s, mentionNum);
				analizer.docMentionMap.put(s+"\t"+partList+ "\t"+mentionNum, uMentions);
			}

			for (int i= 0; i < docList.size(); i++){
				String d = docList.get(i).split("\t")[0];

				if (!docPart.containsKey(d))
					docPart.put(d, 1);
			}
			docList = new ArrayList<String>();
			docList.addAll(analizer.docMentionMap.keySet());

			

			for (int i= 0; i < docList.size()-1; i++){
				for (int j = i+1; j < docList.size(); j++){
					String fid = docList.get(i);
					String sid = docList.get(j);
					String fName = fid.split("\t")[0];
					String sName = sid.split("\t")[0];


					Set<String> common = new HashSet<String>();

					common.addAll(analizer.docMentionMap.get(fid));
					common.retainAll(analizer.docMentionMap.get(sid));
					common.removeAll(veryFrequent);
					double min = (double)Math.min(analizer.docMentionMap.get(fid).size(), analizer.docMentionMap.get(sid).size());
					double ratio = (min > 0 ? common.size()/min: 0);

					if (ratio > 0.05){
						System.out.println(Arrays.toString(common.toArray()) + " " + common.size() + " " + min + " " + ratio);
						if (!similarDocs.containsKey(fName))
							similarDocs.put(fName, new ArrayList<String>());
						if (!similarDocs.containsKey(sName))
							similarDocs.put(sName, new ArrayList<String>());
						similarDocs.get(fName).add(sName);
						similarDocs.get(sName).add(fName);
					}
				}

			}

			Map<String, String> nonRedDocs = new HashMap<String, String>();
			for (String d : docList){
				String name = d.split("\t")[0];
				if (!similarDocs.containsKey(name)){
					nonRedDocs.put(name, d);
				}
			}


			nonRedDocs = docSelection("test_doc_names", nonRedDocs, docPart, similarDocs, docMentionNum);
			nonRedDocs = docSelection("dev_doc_names", nonRedDocs, docPart, similarDocs, docMentionNum);

			PrintWriter pw = new PrintWriter("finalRemaining.txt");
			for (String s : nonRedDocs.keySet())
				pw.println(nonRedDocs.get(s));

			List<String> keys = new ArrayList<String>();
			keys.addAll(similarDocs.keySet());

			for (String s : keys){
				if (similarDocs.containsKey(s))
					for (String d : similarDocs.get(s))
						similarDocs.remove(d);
			}

		    List sorted = new LinkedList<java.util.Map.Entry<String, List<String>>>(similarDocs.entrySet());
		       // Defined Custom Comparator here
		    Collections.sort(sorted, new Comparator() {
		            public int compare(Object o1, Object o2) {
		               return ((Comparable) (((List) (((Map.Entry) (o1)).getValue())).size()))
		                  .compareTo((((List) (((Map.Entry) (o2)).getValue())).size()));
		            }
		       });
		       
		    int allPartSoFar = 0, allDocSoFar = 0;
			for (Iterator it = sorted.iterator(); it.hasNext();){
				Map.Entry entry = (Map.Entry) it.next();
				String s = (String) entry.getKey();
				int partNo = 0, mentionNum = 0;
				for (String s1 : similarDocs.get(s)){
					partNo += docPart.get(s1);
					if (!docMentionNum.containsKey(s1)){
						System.out.println(Arrays.toString(docMentionNum.keySet().toArray()));
						System.out.println(s1);
					}
					mentionNum += docMentionNum.get(s1);
				}
				pw.println(s + " " + Arrays.toString(similarDocs.get(s).toArray()) + " - " + partNo + " " + mentionNum);
			}

			pw.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	static Map<String, String> docSelection(String docNameListFile, Map<String, String> nonRedDocs, 
			Map<String, Integer> docPart, Map<String, List<String>> similarDocs, Map<String, Integer> mentionCount){
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(docNameListFile));

			String line;
			Set<String> acceptable = new HashSet<String>();
			Set<String> unacceptable = new HashSet<String>();


			Set<String> met = new HashSet<String>();

			while( ((line = reader.readLine())!= null)){
				String docName = line.split("\\s+")[2];
				docName = docName.substring(1, docName.indexOf(")"));

				if (met.contains(docName))
					continue;

				if (nonRedDocs.containsKey(docName) && Integer.parseInt(nonRedDocs.get(docName).split("\t")[2]) >=10){
					acceptable.add("+ " + nonRedDocs.get(docName));
					nonRedDocs.remove(docName);
				}
				else{
					unacceptable.add(docName);
				}
				met.add(docName);
			}

			reader.close();
			boolean isResolved = false;
			List<String> resolved = new ArrayList<String>();
			for(String s : unacceptable){
				if (resolved.contains(s))
					continue;
				String genre = s.substring(0, s.indexOf("/"));

				for (String r : nonRedDocs.keySet()){
					//					int p = docPart.get(r);
					int mentionNo = Integer.parseInt(nonRedDocs.get(r).split("\t")[2]);
					if (r.startsWith(genre) && mentionNo >= 10){
						acceptable.add(nonRedDocs.get(r));
						resolved.add(s);
						nonRedDocs.remove(r);
						isResolved = true;
						break;
					}

				}	

				//				int part = docPart.get(s);
//				if (!isResolved){
//
//					if (similarDocs.containsKey(s) && similarDocs.get(s).size() <= 10){
//						int mentionNum = 0;
//						for (String d : similarDocs.get(s)){
//							mentionNum += mentionCount.get(d);
//						}
//						if (mentionNum > (10 * similarDocs.get(s).size())){
//							System.out.println("#### " +similarDocs.get(s).size());
//							acceptable.add(s);
//							resolved.add(s);
//							List<String> toBeRemoved = new ArrayList<String>();
//							toBeRemoved.add(s);
//							for (String d : similarDocs.get(s)){
//								acceptable.add(d);
//								toBeRemoved.add(d);
//								for (String s2 : unacceptable){
//									if (!resolved.contains(s2) && s2.substring(0, s2.indexOf("/")).equals(d.substring(0, d.indexOf("/")))){
//										resolved.add(s2);
//										break;
//									}
//								}
//							}
//
//							for (String r : toBeRemoved)
//								similarDocs.remove(r);
//						}
//					}
//				}

			}

			PrintWriter pw = new PrintWriter(docNameListFile+"_new_split");
			pw.println("Chosen: ");
			for (String s : acceptable)
				pw.println(s);
			pw.println();
			pw.println("Unresolved");
			for (String s : unacceptable)
				if (!resolved.contains(s))
					pw.println(s+"\t"+docPart.get(s));
			pw.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return nonRedDocs;
	}



}
