package edu.stanford.nlp.coref.misc;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.coref.CorefDocumentProcessor;
import edu.stanford.nlp.coref.CorefProperties;
import edu.stanford.nlp.coref.CorefProperties.Dataset;
import edu.stanford.nlp.coref.data.Dictionaries;
import edu.stanford.nlp.coref.data.Document;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.util.StringUtils;

public class MentionIndexGenerator implements CorefDocumentProcessor {
	public PrintWriter pw=null;

	public MentionIndexGenerator() {
		try {
			pw = new PrintWriter("mentionIndices_dev.txt");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	@Override
	public void process(int id, Document document) {
		pw.println(document.conllDoc.documentID);
		for (List<Mention> l : document.predictedMentions)
			for (Mention m : l)
				pw.println(m.sentNum + " - " + m.startIndex + " - " + m.endIndex);
	}

	@Override
	public void finish() throws Exception {pw.close();}

	public static void main(String[] args) throws Exception {
		Properties props = StringUtils.argsToProperties(new String[] {"-props", args[0]});
		Dictionaries dictionaries = new Dictionaries(props);
		CorefProperties.setInput(props, Dataset.DEV);
		new MentionIndexGenerator().run(props, dictionaries);
		
	}
}
