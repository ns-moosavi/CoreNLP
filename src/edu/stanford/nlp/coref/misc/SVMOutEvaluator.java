package edu.stanford.nlp.coref.misc;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class SVMOutEvaluator {

	public static void main(String[] args){
		double tp = 0, fp = 0, tn = 0, fn = 0;
		BufferedReader br2;
		try {
			BufferedReader br1 = new BufferedReader(new FileReader("/data/nlp/moosavne/git/CoreNLP/data_anaphoricity/wikiCoref/svm_test"));

			br2 = new BufferedReader(new FileReader("/data/nlp/moosavne/software/svm_light/wikiTest.out"));
			String line1 = "", line2 = "";
			while ((line1 = br1.readLine()) != null) {
				String[] t = line1.split("\\s+");
				int label = Integer.parseInt(t[0]);
				line2 = br2.readLine();
				double val = Double.parseDouble(line2);
				if (val > 0){
					if (label == 1)
						tp ++;
					else
						fp++;
				}
				else{
					if (label == 1)
						fn ++;
					else
						tn++;
					
				}
			}
			
			double ana_recall = tp/(tp+fn);
			double ana_pre = tp /(tp+fp);
			double ana_f = 2*ana_recall * ana_pre / (ana_pre+ana_recall);
			double rec = tn/(tn+fp);
			double pre = tn/(tn+fn);
			double f = 2*rec*pre/(rec+pre);
			
			System.out.println("Anaphoric");
			System.out.println(ana_recall + " " + ana_pre + " " + ana_f);
			System.out.println("Non-anaphoric");
			System.out.println(rec + " " + pre + " " + f);
			br1.close();
			br2.close();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}


		
	}
}
