package edu.stanford.nlp.coref.neural;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.coref.data.Dictionaries;
import edu.stanford.nlp.coref.data.Dictionaries.MentionType;
import edu.stanford.nlp.coref.data.Document;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.util.Pair;

public class CortFeatureFileParser {
	String nomFileName = "/data/nlp/moosavne/git/featureMiner/null";
	String namFileName = "/data/nlp/moosavne/git/featureMiner/null";
	String proFileName = "/data/nlp/moosavne/git/featureMiner/null";

	public final List<Set<String>> m_nomFeatures;
	public final List<Set<String>> m_namFeatures;
	public final List<Set<String>> m_proFeatures;

	public CortFeatureFileParser(){
		m_nomFeatures = readFeatureFile(nomFileName);
		m_namFeatures = readFeatureFile(namFileName);
		m_proFeatures = readFeatureFile(proFileName);

	}

	private List<Set<String>> readFeatureFile(String fileName){
		List<Set<String>> features = new ArrayList<>();

		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			String line;

			while( ((line = reader.readLine())!= null)){ 
				String[] attVals = line.split("\\+");
				Set<String> attValSet = new HashSet<>();
				for (String s : attVals)
					attValSet.add(s);

				features.add(attValSet);
			}

			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

		return features;
	}

	public boolean[] getFeatures(Set<String> ana_features, Set<String> ant_features, Set<String> pairFeatures, String identifier){
		boolean[] features = null;
		List<Set<String>> featureList = null;
		boolean isViable = false;

		if (identifier.equals("NAM")){
			features = new boolean[m_namFeatures.size()];
			featureList = m_namFeatures;
			isViable = true;
		}
		else if (identifier.equals("NOM")){
			features = new boolean[m_nomFeatures.size()];
			featureList = m_nomFeatures;
			isViable = true;
		}
		else if (identifier.equals("PRO")){
			features = new boolean[m_proFeatures.size()];
			featureList = m_proFeatures;
			isViable = true;
		}

		if (isViable){
			Set<String> allAttr = new HashSet<>();
			allAttr.addAll(ana_features);
			allAttr.addAll(ant_features);
			allAttr.addAll(pairFeatures);

			for (int i = 0; i < featureList.size(); i++){
				if (allAttr.containsAll(featureList.get(i)))
					features[i] = true;
				else
					features[i] = false;
			}
		}
		return features;
	}

	Set<String> getMentionFeatures(Mention m, String identifier){
		Set<String> atts = new HashSet<String>();
		String prefix = "";

		if (identifier.equalsIgnoreCase("ana"))
			prefix = "ana_";
		else
			prefix = "ant_";

		atts.add(prefix +"type="+m.mentionType.toString());
		atts.add(prefix+"fine_type="+fine_type(m));
		atts.add(prefix+"gender="+m.gender.toString());
		atts.add(prefix+"number="+m.number.toString());
		atts.add(prefix+"head_ner="+m.nerString);
		atts.add(prefix+"length="+getBinnedValue(m.originalSpan.size()));
		Pair<IndexedWord, String> verbDependency = Mention.findDependentVerb(m);
		String dep = verbDependency.second();
		if (dep != null){
			atts.add(prefix +"deprel="+dep);
//			atts.add(prefix+"gov="+verbDependency.first.lemma());
		}
		else{
			atts.add(prefix +"deprel="+"null");
//			atts.add(prefix+"gov="+"null");
		}

		return atts;
	}

	Set<String> getPairwiseFeatures(Document document, Mention m, Mention ante, Dictionaries dict, 
			boolean[] firstCompatibles){
		Set<String> res = new HashSet<>();
		boolean exact_match = false, head_match=false,token_contained = false, head_contained = false;


		if (m.spanToString().equalsIgnoreCase(ante.spanToString()))
			exact_match = true;
		if (m.headString.equalsIgnoreCase(ante.headString))
			head_match = true;
		else{
			if (m.spanToString().toLowerCase().contains(ante.spanToString().toLowerCase()) || ante.spanToString().toLowerCase().contains(m.spanToString().toLowerCase()))
				token_contained = true;
			if (!token_contained && !head_contained && 
					m.spanToString().toLowerCase().contains(ante.headString.toLowerCase()) || ante.spanToString().toLowerCase().contains(m.headString.toLowerCase()))
				head_contained = true;
		}
		res.add("exact_match="+(exact_match? "True" : "False"));
		res.add("head_match="+(head_match? "True" : "False"));
		res.add("token_contained="+(token_contained? "True" : "False"));
		res.add("head_contained="+(head_contained? "True" : "False"));
		res.add("same_speaker="+((m.speakerInfo != null && m.speakerInfo == ante.speakerInfo)? "True" : "False"));
		res.add("same_number="+((m.numbersAgree(ante, true))? "True" : "False"));
		res.add("compatible_number="+((m.numbersAgree(ante, false))? "True" : "False"));
		res.add("same_gender="+((m.gendersAgree(ante, true))? "True" : "False"));
		res.add("compatible_gender="+((m.gendersAgree(ante, false))? "True" : "False"));
		res.add("same_animacy="+((m.animaciesAgree(ante, true))? "True" : "False"));
		res.add("compatible_animacy="+((m.animaciesAgree(ante, false))? "True" : "False"));
		boolean compatibleGenderNumber = false;
		if (m.numbersAgree(ante) && m.gendersAgree(ante)){
			compatibleGenderNumber = true;
		}
		res.add("comp_gen_num="+(compatibleGenderNumber? "True" : "False"));
		res.add("f_comp_obj="+((compatibleGenderNumber && !firstCompatibles[0] && ante.isDirectObject)? "True" : "False"));
		res.add("f_comp_subj="+((compatibleGenderNumber && !firstCompatibles[1] && ante.isSubject)? "True" : "False"));
		res.add("f_com_preObj="+((compatibleGenderNumber && !firstCompatibles[2] && ante.isPrepositionObject)? "True" : "False"));
		res.add("att_agree="+((m.attributesAgree(ante, dict))? "True" : "False"));
		res.add("sentence_distance="+getBinnedValue(m.sentNum - ante.sentNum));

		return res;
	}

	private String getBinnedValue(int val){

		int[] vals = {0, 1, 2, 3};
		for (Integer v : vals){
			if (val == v)
				return ""+val;
		}
		if (val <= 5)
			return "<=5";
		return ">5";
	}


	private String fine_type(Mention m){
		if (m.mentionType == MentionType.NOMINAL){
			if (Pattern.matches("^(the|this|that|these|those|my|your|his|her|its|our|their)", m.originalSpan.get(0).word().toLowerCase()))
				return "DEF";
			if (m.headWord.get(CoreAnnotations.PartOfSpeechAnnotation.class).startsWith("NNP"))
				return "DEF";
			else return "INDEF";
		}
		else if (m.mentionType == MentionType.PRONOMINAL){
			String pronoun = m.originalSpan.get(0).word().toLowerCase();
			if (Pattern.matches("^(he|him|himself|his)$",  pronoun))
				return "he";
			else if (Pattern.matches("^(she|her|herself|hers|her)$",  pronoun))
				return "she";
			else if (Pattern.matches("^(it|itself|its)$",  pronoun))
				return "it";
			else if (Pattern.matches("^(they|them|themselves|theirs|their)$",  pronoun))
				return "they";
			else if (Pattern.matches("^(i|me|myself|mine|my)$",  pronoun))
				return "i";
			else if (Pattern.matches("^(you|yourself|yourselves|yours|your)$",  pronoun))
				return "you";
			else if (Pattern.matches("^(we|us|ourselves|ours|our)$",  pronoun))
				return "we";
		}
		else if (m.mentionType == MentionType.PROPER)
			return "NAM";

		return "Other";

	}
}
