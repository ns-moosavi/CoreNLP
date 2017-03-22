package edu.stanford.nlp.coref.misc;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class SVMAttentionErrorAnalysis {

	public static void main(String[] args){

		try {
			boolean isWiki = false;
			BufferedReader svmLabel = new BufferedReader(new FileReader("/data/nlp/moosavne/git/CoreNLP/data_anaphoricity/deep_coref_svm/test_ids")); //svm_wiki_test_ids
			BufferedReader svmOUt = new BufferedReader(new FileReader("/data/nlp/moosavne/git/CoreNLP/data_anaphoricity/deep_coref_svm/test_scores")); //wikiTest.out
			BufferedReader attLabel = new BufferedReader(new FileReader("/data/nlp/moosavne/git/CoreNLP/no_linguist_in_lstm_no_proper_dropping.txt")); //attentionWikiOut.txt
			BufferedReader attSamples = new BufferedReader(new FileReader("/data/nlp/moosavne/git/CoreNLP/data_anaphoricity/wikiCoref/test"));//wikiNNAnaphoricityData_withDocId.txt
			PrintWriter pw = new PrintWriter("/data/nlp/moosavne/git/CoreNLP/svmAttDiff.txt");
			PrintWriter pw1 = new PrintWriter("/data/nlp/moosavne/git/CoreNLP/attSVMDiff.txt");

			Map<String, String> posIdToSamples = new HashMap<>();
			Map<String, String> negIdToSamples = new HashMap<>();
			Map<String, Double> svmTP = new HashMap<>();
			Map<String, Double> attTP = new HashMap<>();
			Map<String, Double> svmFN = new HashMap<>();
			Map<String, Double> attFN = new HashMap<>();
			Map<String, Double> svmFP = new HashMap<>();
			Map<String, Double> attFP = new HashMap<>();


			String line1 = "";
			while ((line1 = attSamples.readLine()) != null) {
				String[] s = line1.split(" QQQQQQQQQQ ");
				double label = Double.parseDouble(s[s.length-1]);
				if (label  == 1)
					posIdToSamples.put(s[0].trim()+ " QQQQQQQQQQ "+s[1].trim(), line1);
				else
					negIdToSamples.put(s[0].trim()+ " QQQQQQQQQQ "+s[1].trim(), line1);
			}
			attSamples.close();

			int sTP = 0, sFP = 0, sTN = 0, sFN = 0;
			int[][][] counts = new int[2][4][4];

			System.out.println(posIdToSamples.size() + " vs " + negIdToSamples.size());
			while ((line1 = svmLabel.readLine()) != null) {
				line1 = line1.trim();
				double svmVal = Double.parseDouble(svmOUt.readLine());
				if (posIdToSamples.keySet().contains(line1.trim())){
					int type = posIdToSamples.get(line1).contains("PRONOMINAL") ? 2 : (posIdToSamples.get(line1).contains(" NOMINAL") ? 1 : (posIdToSamples.get(line1).contains(" PROPER") ? 0 : 3));

					if (svmVal > 0){
						sTP++;
						counts[0][type][0]++;
					}
					else {
						sFN++;
						counts[0][type][1]++;
					}
				}
				else if (negIdToSamples.keySet().contains(line1.trim())){
					int type = negIdToSamples.get(line1).contains("PRONOMINAL") ? 2 : (negIdToSamples.get(line1).contains(" NOMINAL") ? 1 : (negIdToSamples.get(line1).contains(" PROPER") ? 0 : 3));

					if (svmVal > 0){
						sFP++;
						counts[0][type][2]++;
						svmFP.put(line1.trim(), svmVal);
					}
					else {
						sTN++;
						counts[0][type][3]++;
					}
				}
				else
					System.out.println("?? " + line1);

				if (svmVal > 0 && posIdToSamples.keySet().contains(line1.trim()))
					svmTP.put(line1.trim(), svmVal);
				if (svmVal < 0 && posIdToSamples.keySet().contains(line1.trim()))
					svmFN.put(line1.trim(), svmVal);
			}

			double sRecall = sTP /(double)(sTP+sFN);
			double sPre = sTP/(double)(sTP+sFP);
			double sF = 2*sRecall * sPre / (sRecall+sPre);
			double sNRecall = sTN/(double)(sTN+sFP);
			double sNPre = sTN/(double)(sTN+sFN);
			System.out.println("TP: " + sTP + " TN: " + sTN + " FP: "+ sFP + " FN: " + sFN);
			System.out.println("SVM recall: " + sRecall + " precision: " + sPre + " F1: " +sF );
			System.out.println("SVM non-anaphoric recall: " + sNRecall + " precision: " + sNPre + " F1: " +(2*sNRecall*sNPre)/(sNRecall+sNPre) );

			svmLabel.close();
			svmOUt.close();
			int common = 0, missing = 0;
			int aTP = 0, aFP = 0, aTN = 0, aFN = 0;
			double thresold = 0.4;
			while ((line1 = attLabel.readLine()) != null) {
				double attScore;
				String id;
				if (isWiki){
					String fName = line1.substring(0, line1.indexOf("txt")+3);
					String rest = line1.substring(line1.indexOf("txt")+3).trim();
					String[] s = rest.split("\\s+");
					id = fName + " QQQQQQQQQQ "+s[0].trim();
					//				System.out.println(Arrays.toString(s));
					attScore = Double.parseDouble(s[2].substring(0, s[2].indexOf("]")));
				}
				else{
					String[] s = line1.split("\\s+");
					id = s[0].trim() + " QQQQQQQQQQ "+s[1].trim();
					attScore = Double.parseDouble(s[3].substring(0, s[3].indexOf("]")));
				}

				if (posIdToSamples.keySet().contains(id)){
					int type = posIdToSamples.get(id).contains("PRONOMINAL") ? 2 : (posIdToSamples.get(id).contains(" NOMINAL") ? 1 : (posIdToSamples.get(id).contains(" PROPER") ? 0 : 3));

					if (attScore >= thresold){
						aTP++;
						attTP.put(id, attScore);
						counts[1][type][0]++;
					}
					else{
						attFN.put(id, attScore);
						aFN++;
						counts[1][type][1]++;
					}
				}
				else if (negIdToSamples.keySet().contains(id)){
					int type = negIdToSamples.get(id).contains("PRONOMINAL") ? 2 : (negIdToSamples.get(id).contains(" NOMINAL") ? 1 : (negIdToSamples.get(id).contains(" PROPER") ? 0 : 3));

					if (attScore >= thresold){
						aFP++;
						counts[1][type][2]++;
						attFP.put(id, attScore);
					}
					else{
						aTN++;
						counts[1][type][3]++;
					}
				}
				else
					System.out.println("!!!! "+ id);

				if (posIdToSamples.keySet().contains(id) && svmTP.containsKey(id)){
					if (attScore >= thresold){
						//						pw.println("!!COMMON " + posIdToSamples.get(id) + " svm: " + svmTP.get(id) + " att: " + attScore);
						common++;
					}
					else {
						pw.println("$$MISS " + posIdToSamples.get(id)+ " svm: " + svmTP.get(id) + " att: " + attScore);
						missing++;
					}

				}
			}

			for (String id : attTP.keySet()){
				if (!svmTP.containsKey(id))
					pw1.println("$$MISS " + posIdToSamples.get(id)+ " svm: " + svmTP.get(id) + " att: " + attTP.get(id));

			}

			pw1.close();
			
			pw1 = new PrintWriter("/data/nlp/moosavne/git/CoreNLP/svmWorseProper");
			
			for (String id : svmTP.keySet()){
				if (!attTP.containsKey(id))
					if (posIdToSamples.get(id).contains(" PROPER "))
						pw1.println(posIdToSamples.get(id) + " att: " + svmTP.get(id) + " svm: " + attFN.get(id) );
			}
//			pw1.close();
//			pw1 = new PrintWriter("/data/nlp/moosavne/git/CoreNLP/svmCorrectFN");
//			for (String id : attFN.keySet()){
//				if (!svmFN.containsKey(id))
//					pw1.println("SVMGood " + posIdToSamples.get(id));
//			}
//			pw1.close();
//
//			pw1 = new PrintWriter("/data/nlp/moosavne/git/CoreNLP/svmFP");
//			for (String id : svmFP.keySet()){
//				if (!attFN.containsKey(id))
//					pw1.println("SVMBAD! " + negIdToSamples.get(id));
//			}
//			pw1.close();

//			pw1 = new PrintWriter("/data/nlp/moosavne/git/CoreNLP/attFP");
//			for (String id : attFP.keySet()){
//				if (!svmFN.containsKey(id))
//					pw1.println("ATTBAD! " + negIdToSamples.get(id));
//			}
//			pw1.close();
			double aRecall = aTP /(double)(aTP+aFN);
			double aPre = aTP/(double)(aTP+aFP);
			double aF = 2*aRecall * aPre / (aRecall+aPre);
			double aNRecall = aTN/(double)(aTN+aFP);
			double aNPre = aTN/(double)(aTN+aFN);

			System.out.println("TP: " + aTP + " TN: " + aTN + " FP: "+ aFP + " FN: " + aFN);
			System.out.println("Attention recall: " + aRecall + " precision: " + aPre + " F1: " +aF );
			System.out.println("Attention non-anaphoric recall: " + aNRecall + " precision: " + aNPre + " F1: " +(2*aNRecall*aNPre)/(aNRecall+aNPre) );

			System.out.println("Common: " + common + " Missing: " + missing);

			for (int i = 0; i < 4; i++){
				String iden = (i==0 ? "Proper " : (i==1 ? " Nominal " : (i==2 ? "Pronoun " : "List ")));
				System.out.println(iden);
				for (int k = 0; k < 2; k++){
					String model = (k == 0 ? " SVM " : " Attention ");
					System.out.println(model);
					double recal = counts[k][i][0] / (double)(counts[k][i][0]+counts[k][i][1]);
					double prec = counts[k][i][0] / (double)(counts[k][i][0]+counts[k][i][2]);
					double f = 2 * recal * prec / (recal+prec);
					System.out.println("Recall: " + recal + " Precision: " + prec + " F1: " + f);
				}
			}
			attLabel.close();
			pw.close();

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
}
