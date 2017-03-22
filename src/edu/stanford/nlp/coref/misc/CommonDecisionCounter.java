package edu.stanford.nlp.coref.misc;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class CommonDecisionCounter {
	public static void main(String[] args){
		try {
			BufferedReader br = new BufferedReader(new FileReader("/data/nlp/moosavne/git/CoreNLP/gold_dev_non_anaphoric_it.txt"));
			BufferedReader br1 = new BufferedReader(new FileReader("/data/nlp/moosavne/git/CoreNLP/deep_coref_dev_non_anaphoric_it.txt"));
			List<String> gold = new ArrayList<String>();
			List<String> predicted = new ArrayList<String>();
			String line = "";
			while((line = br.readLine()) != null){
				String[] split = line.split("\\s+");
				gold.add(split[0]+"   " + split[1] + "   " + split[2]+ "   " + split[3] + "   " + split[4]+ "   " + split[5]);
			}
			br.close();
			while((line = br1.readLine()) != null){
				String[] split = line.split("\\s+");
				predicted.add(split[0]+"   " + split[1] + "   " + split[2]+ "   " + split[3] + "   " + split[4]+ "   " + split[5]);
			}
			br1.close();
			
			int detected = 0;
			for (String s : gold){
				if (predicted.contains(s))
					detected++;
			}
			
			for (String s : predicted)
				if (gold.contains(s))
					System.out.println(s);
			System.out.println("All non-anaphoric it: " + gold.size()+ " , all detected non-anaphoric it: " + predicted.size());
			
			System.out.println("Correctly detected non-anaphoric it: " + detected);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
