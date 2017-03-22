package edu.stanford.nlp.coref.misc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CoNLLGenreSpliter {
	
	public static void main(String[] args){
	try {
		List<String> genres =Arrays.asList("bn", "bc", "wb", "mz", "tc", "nw", "pt");
		List<String> datasets = Arrays.asList("train", "development", "test");
		
		PrintWriter train_pw, dev_pw, test_pw;
		String outPath = "/data/nlp/moosavne/corpora/conll12/splits/";
		String trainFileName = "/data/nlp/moosavne/corpora/conll-2012/all_eng_train_v4_auto_conll";
		String devFileName = "/data/nlp/moosavne/corpora/conll-2012/all_eng_dev_v4_auto_conll";
		String testFileName = "/data/nlp/moosavne/corpora/conll-2012/all_eng_test_v4_ordered_auto_conll.gold";
		BufferedReader trainReader = new BufferedReader(new FileReader(trainFileName));
		BufferedReader devReader = new BufferedReader(new FileReader(devFileName));
		BufferedReader testReader = new BufferedReader(new FileReader(testFileName));

	
		for (String genre : genres){
			trainReader = new BufferedReader(new FileReader(trainFileName));
			devReader = new BufferedReader(new FileReader(devFileName));
			testReader = new BufferedReader(new FileReader(testFileName));
			
			File dir = new File(outPath+"/"+genre);
			if (!dir.exists())
				dir.mkdir();
			dir = new File(outPath+"/"+genre+"/data");
			if (!dir.exists())
				dir.mkdir();
			List<String> outNames = new ArrayList<String>();
			for (String d : datasets){
				dir = new File(outPath+"/"+genre+"/data/"+d);
				if (!dir.exists())
					dir.mkdir();
				outNames.add(dir+"/"+d+"_"+genre+"_auto_conll");
			}
					
			System.out.println(Arrays.toString(outNames.toArray()));
			train_pw = new PrintWriter(outNames.get(0));
			dev_pw = new PrintWriter(outNames.get(1));
			test_pw = new PrintWriter(outNames.get(2));
			
			String line = "";
			boolean prnt = false;
			while((line = trainReader.readLine())!= null){
				if (line.startsWith("#begin document")){
					String name = line.substring(line.indexOf("(")+1, line.indexOf(")"));
					
					if (!name.startsWith(genre))
						prnt = true;
					else
						prnt = false;
				}
				if (prnt)
					train_pw.println(line);
			}
			prnt = false;
			while((line = devReader.readLine())!= null){
				if (line.startsWith("#begin document")){
					String name = line.substring(line.indexOf("(")+1, line.indexOf(")"));
					
					if (!name.startsWith(genre))
						prnt = true;
					else
						prnt = false;
				}
				if (prnt)
					dev_pw.println(line);
			}
			prnt = false;
			while((line = testReader.readLine())!= null){
				if (line.startsWith("#begin document")){
					String name = line.substring(line.indexOf("(")+1, line.indexOf(")"));
					
					if (name.startsWith(genre))
						prnt = true;
					else
						prnt = false;
				}
				if (prnt)
					test_pw.println(line);
			}
			train_pw.close();
			dev_pw.close();
			test_pw.close();
			trainReader.close();
			testReader.close();
			devReader.close();
		}


		trainReader.close();
		testReader.close();
		devReader.close();
		
		
	} catch (Exception e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
	
	
}
}
