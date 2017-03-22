package edu.stanford.nlp.dcoref.sievepasses;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

import edu.stanford.nlp.dcoref.Dictionaries;
import edu.stanford.nlp.dcoref.Dictionaries.MentionType;
import edu.stanford.nlp.dcoref.Document;
import edu.stanford.nlp.dcoref.Mention;
import edu.stanford.nlp.dcoref.Rules;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.util.Pair;

public class FeatureFileParser {
		String featureFileName = "/data/nlp/moosavne/git/featureMiner/dev_1p_5_60_p";


		public final List<Set<String>> m_allFeatures;

		public FeatureFileParser(){
			m_allFeatures = readFeatureFile(featureFileName);

		}

		private List<Set<String>> readFeatureFile(String fileName){
			List<Set<String>> features = new ArrayList<>();

			try {
				BufferedReader reader = new BufferedReader(new FileReader(fileName));
				String line;

				while( ((line = reader.readLine())!= null)){ 
					line = line.split("\\s+")[0];
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

		public boolean[] getFeatures(Set<String> ana_features, Set<String> ant_features, Set<String> pairFeatures){
			boolean[] features = new boolean[m_allFeatures.size()];


			Set<String> allAttr = new HashSet<>();
			allAttr.addAll(ana_features);
			allAttr.addAll(ant_features);
			allAttr.addAll(pairFeatures);

			for (int i = 0; i < m_allFeatures.size(); i++){
				if (allAttr.containsAll(m_allFeatures.get(i))){
					features[i] = true;
//					System.out.println(Arrays.toString(allAttr.toArray()) + " -- " + Arrays.toString(m_allFeatures.get(i).toArray()));
				}
				else
					features[i] = false;
			}

			return features;
		}

		public int hasPositiveFeature(Set<String> ana_features, Set<String> ant_features, Set<String> pairFeatures){

			Set<String> allAttr = new HashSet<>();
			allAttr.addAll(ana_features);
			allAttr.addAll(ant_features);
			allAttr.addAll(pairFeatures);

			for (int i = 0; i < m_allFeatures.size(); i++){
				if (allAttr.containsAll(m_allFeatures.get(i))){
					return i;
				}
			}

			return -1;
		}
		
		public Set<String> getMentionFeatures(Mention m, String identifier){
			Set<String> atts = new HashSet<String>();
			String prefix = "";

			if (identifier.equalsIgnoreCase("ana"))
				prefix = "ana_";
			else
				prefix = "ant_";

		    Pair<IndexedWord, String> verbDependency = Mention.findDependentVerb(m);
		    String depRelation = verbDependency.second() != null ? verbDependency.second() :"no-parent";

			atts.add(prefix +"type="+m.mentionType.toString());
			atts.add(prefix+"f_type="+fine_type(m));
			atts.add(prefix+"ner="+m.nerString);
			atts.add(prefix+"len="+getBinnedValue(m.originalSpan.size()));
			atts.add(prefix +"rel="+depRelation);
			atts.add(prefix +"f_POS="+m.sentenceWords.get(m.startIndex).get(CoreAnnotations.PartOfSpeechAnnotation.class));
			atts.add(prefix + "l_POS="+m.sentenceWords.get(m.endIndex-1).get(CoreAnnotations.PartOfSpeechAnnotation.class));
			atts.add(prefix + "head_POS="+m.sentenceWords.get(m.headIndex).get(CoreAnnotations.PartOfSpeechAnnotation.class));
			atts.add(prefix + "n_POS="+(m.endIndex < m.sentenceWords.size() ? m.sentenceWords.get(m.endIndex).get(CoreAnnotations.PartOfSpeechAnnotation.class) : "END"));
			atts.add(prefix + "nn_POS="+(m.endIndex+1 < m.sentenceWords.size() ? m.sentenceWords.get(m.endIndex+1).get(CoreAnnotations.PartOfSpeechAnnotation.class) : "END"));
			atts.add(prefix + "p_POS="+(m.startIndex-1 > 0 ? m.sentenceWords.get(m.startIndex-1).get(CoreAnnotations.PartOfSpeechAnnotation.class) : "START"));
			atts.add(prefix +  "pp_POS="+ (m.startIndex-2 > 0 ? m.sentenceWords.get(m.startIndex-2).get(CoreAnnotations.PartOfSpeechAnnotation.class) : "START"));
			
			return atts;
		}

		public static String fine_type(Mention m){
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
		public List<Set<String>> getPairwiseFeatures(Document document, Mention m, List<Mention> allAnte, Dictionaries dict){
			List<Set<String>> allResults = new ArrayList<Set<String>>();
			boolean[] firstCompatibles = new boolean[4];

			for (int i = allAnte.size()-1; i >=0; i--){

				Mention ante = allAnte.get(i);			
				Pair<IndexedWord, String> verbDependency = Mention.findDependentVerb(m);
			    String ant_depRelation = verbDependency.second() != null ? verbDependency.second() :"no-parent";
				
				Set<String> res = new HashSet<>();
				boolean exact_match = false, head_match=false,token_contained = false, head_contained = false;

				if (m.spanToString().equalsIgnoreCase(ante.spanToString()))
					exact_match = true;
				
				if (!m.spanToString().equalsIgnoreCase(ante.spanToString())){
					if (m.spanToString().toLowerCase().contains(ante.spanToString().toLowerCase()) || ante.spanToString().toLowerCase().contains(m.spanToString().toLowerCase()))
						token_contained = true;
					if (m.headString.equalsIgnoreCase(ante.headString)){
						head_match = true;
					}

					if (!token_contained && !head_contained && 
							m.spanToString().toLowerCase().contains(ante.headString.toLowerCase()) || ante.spanToString().toLowerCase().contains(m.headString.toLowerCase()))
						head_contained = true;
				}
				res.add("exact_match="+(exact_match? "True" : "False"));
				res.add("head_match="+(head_match? "True" : "False"));
				res.add("token_contained="+(token_contained? "True" : "False"));
				res.add("head_contained="+(head_contained? "True" : "False"));
				res.add("compatible_modifiers="+(m.compatibleModifiers(ante) ? "True" : "False"));
				if (head_match && !firstCompatibles[0] && m.compatibleModifiers(ante)){
					res.add("first_compatible_head_match=True");
					firstCompatibles[0] = true;
				}
				if (m.numbersAgree(ante))
					res.add("comp_number=True");

				if (m.gendersAgree(ante))
					res.add("comp_gender=True");


				if (m.animaciesAgree(ante))
					res.add("comp_animacy=True");

				boolean compatible = false;
				if (m.numbersAgree(ante) && m.gendersAgree(ante) && m.animaciesAgree(ante)){
					res.add("comp_attributes=True");
					compatible = true;
				}


				if (compatible && !firstCompatibles[1]){
					res.add("f_comp_att=True");
					firstCompatibles[1] = true;
				}
				
				if (!firstCompatibles[2] && (ant_depRelation.startsWith("dobj") || ant_depRelation.startsWith("iobj"))){
					res.add("f_com_obj=True");
					firstCompatibles[2] =true;
				}

				if (!firstCompatibles[3] && ant_depRelation.startsWith("nsubj")){
					res.add("f_comp_subj=True");
					firstCompatibles[3] = true;
				}
				
				if (Rules.isAcronym(m.originalSpan, ante.originalSpan))
					res.add("acronym=True");

				res.add("distance="+getBinnedValue(m.sentNum - ante.sentNum));
				res.add("genre="+document.conllDoc.getDocumentID().split("/")[0]);



				allResults.add(res);
			}
			return allResults;
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

	}
