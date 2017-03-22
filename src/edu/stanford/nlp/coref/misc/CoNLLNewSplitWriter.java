package edu.stanford.nlp.coref.misc;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;


public class CoNLLNewSplitWriter {

	public static void main(String[] args){
		try {
			PrintWriter train = new PrintWriter("train_my_new_split_auto_conll");
			PrintWriter dev = new PrintWriter("dev_my_new_split_auto_conll");
			PrintWriter test = new PrintWriter("test_my_new_split_auto_conll");
			
			BufferedReader trainReader = new BufferedReader(new FileReader("/data/nlp/moosavne/corpora/conll-2012/all_eng_train_v4_auto_conll"));
			BufferedReader devReader = new BufferedReader(new FileReader("/data/nlp/moosavne/corpora/conll-2012/all_eng_dev_v4_auto_conll"));
			BufferedReader testReader = new BufferedReader(new FileReader("/data/nlp/moosavne/corpora/conll-2012/all_eng_test_v4_ordered_auto_conll.gold"));

			BufferedReader devNameReader = new BufferedReader(new FileReader("my_chosen_dev_set"));
			BufferedReader testNameReader = new BufferedReader(new FileReader("my_chosen_test_set"));
			
			List<String> devDocNames = new ArrayList<String>();
			List<String> testDocNames = new ArrayList<String>();
			
			String line = "";
			while( ((line = devNameReader.readLine())!= null)){
				devDocNames.add(line.split("\t")[0].trim());
			}
			
			while( ((line = testNameReader.readLine())!= null)){
				testDocNames.add(line.split("\t")[0].trim());
			}

			PrintWriter targetFile= train;
			while((line = trainReader.readLine())!= null){
				if (line.startsWith("#begin document")){
					String name = line.substring(line.indexOf("(")+1, line.indexOf(")"));
					if (testDocNames.contains(name))
						targetFile = test;
					else if (devDocNames.contains(name))
						targetFile = dev;
					else
						targetFile = train;
					
				}
				targetFile.println(line);
			}
			
			while((line = devReader.readLine())!= null){
				if (line.startsWith("#begin document")){
					String name = line.substring(line.indexOf("(")+1, line.indexOf(")"));
					if (testDocNames.contains(name))
						targetFile = test;
					else if (devDocNames.contains(name))
						targetFile = dev;
					else
						targetFile = train;
					
				}
				targetFile.println(line);
			}
			
			while((line = testReader.readLine())!= null){
				if (line.startsWith("#begin document")){
					String name = line.substring(line.indexOf("(")+1, line.indexOf(")"));
					if (testDocNames.contains(name))
						targetFile = test;
					else if (devDocNames.contains(name))
						targetFile = dev;
					else
						targetFile = train;
					
				}
				targetFile.println(line);
			}

			devNameReader.close();
			testNameReader.close();
			train.close();
			test.close();
			dev.close();
			trainReader.close();
			testReader.close();
			devReader.close();
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}
}
