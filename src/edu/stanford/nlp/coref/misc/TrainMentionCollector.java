package edu.stanford.nlp.coref.misc;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.coref.CorefDocumentProcessor;
import edu.stanford.nlp.coref.CorefUtils;
import edu.stanford.nlp.coref.data.CorefCluster;
import edu.stanford.nlp.coref.data.Dictionaries.MentionType;
import edu.stanford.nlp.coref.data.Document;
import edu.stanford.nlp.coref.data.Mention;

public class TrainMentionCollector  implements CorefDocumentProcessor {
	public Set<String> allGoldMentionStr = new HashSet<String>();
	public Set<String> allMentionStr = new HashSet<String>();

	@Override
	public void process(int id, Document document) {
		for (CorefCluster gold : document.goldCorefClusters.values()) {
			for (Mention m : gold.corefMentions) {
				if (m.mentionType != MentionType.PRONOMINAL)
					allGoldMentionStr.add(m.spanToString().toLowerCase());
			}
		}	
		List<Mention> mentionsList = CorefUtils.getSortedMentions(document);
		for (int i = 0; i < mentionsList.size(); i++) {
			if (mentionsList.get(i).mentionType != MentionType.PRONOMINAL)
				if (!allGoldMentionStr.contains(mentionsList.get(i).spanToString()) )
					allMentionStr.add(mentionsList.get(i).spanToString().toLowerCase());
		}
	}

	@Override
	public void finish() throws Exception {
		// TODO Auto-generated method stub	
	}
}
