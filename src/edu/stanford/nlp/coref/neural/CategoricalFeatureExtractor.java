package edu.stanford.nlp.coref.neural;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.Collections;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.coref.CorefProperties;
import edu.stanford.nlp.coref.CorefRules;
import edu.stanford.nlp.coref.data.Dictionaries;
import edu.stanford.nlp.coref.data.Dictionaries.MentionType;
import edu.stanford.nlp.coref.data.Document;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.NeuralUtils;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.Pair;

/**
 * Extracts string matching, speaker, distance, and document genre features from mentions.
 * @author Kevin Clark
 */
public class CategoricalFeatureExtractor {
	private final Dictionaries dictionaries;
	private final Map<String, Integer> genres;
	private final boolean conll;
	public FeatureFileParser parser;
	public int[] counts;
	public int prunThreshold = 10;
	public static int namFeatureNum = 39;
	public List<String> importantAtts_dev_10_6_60 = Arrays.asList("ana_type=PRONOMINAL", "ant_f_type=she", "ana_f_type=she", "ana_rel=iobj", "ana_ner=EVENT", "first_compatible_head_match=True", 
			"ant_f_type=you", "ana_f_type=we", "ant_rel=nmod:with", "ant_f_type=we", "ana_f_type=i", "ana_rel=no-parent", "ant_f_type=i", "ant_rel=nmod:poss", "ana_rel=nmod:in", 
			"ant_ner=PERSON", "ana_rel=nmod:of", "ant_rel=nmod:of", "ana_rel=nmod:poss", "ana_f_type=it", "ana_rel=compound", "ana_f_type=you", "ana_rel=nsubjpass", "ant_f_type=it", 
			"ana_f_type=he", "ana_f_type=they", "ant_f_type=they", "ant_f_type=he", "ant_f_type=Other", "token_contained=True", "ant_rel=nmod:in", "ana_ner=GPE", "ana_ner=ORG", 
			"head_contained=True", "ant_rel=nsubjpass", "ant_len=3", "ant_len=<=5", "ana_len=<=5", "f_comp_att=True", "ana_len=1", "comp_gender=True", "ant_rel=dobj", "ana_rel=dobj", 
			"ant_len=2", "ant_len=>5", "ana_len=>5", "ant_f_type=DEF", "ant_f_type=INDEF", "ana_type=PROPER", "ana_f_type=NAM", "ana_f_type=DEF", "ant_rel=nsubj", "ana_len=2", 
			"ant_type=PRONOMINAL", "ant_rel=compound", "ant_rel=appos", "ana_rel=appos", "ant_type=LIST", "ant_ner=ORG", "ana_ner=PERSON", "ana_len=3", "ant_ner=GPE", "ana_rel=nsubj", 
			"ant_f_type=NAM", "ana_ner=O", "compatible_modifiers=True", "head_match=True", "ant_type=PROPER", "ant_len=1", "ant_ner=O", "comp_number=True", "comp_animacy=True" , 
			"comp_attributes=True", "ana_type=NOMINAL", "ant_type=NOMINAL");
	
	public List<String> importantAtts_train_20_6_60 = Arrays.asList("comp_attributes=True", "ana_rel=nmod:poss", "ant_rel=nmod:of", "ana_rel=nmod:of", "ant_len=2", 
			"ant_rel=dobj", "ant_len=<=5", "ana_rel=dobj", "ant_len=3", "token_contained=True", "ant_rel=nmod:in", "ant_rel=nmod:poss", "head_match=True", "ana_rel=nsubjpass", 
			"ant_rel=nsubjpass", "ant_rel=nmod:on", "ant_rel=nmod:with", "ana_rel=appos", "ant_ner=NORP", "ant_rel=nmod:from", "ana_rel=compound", 
			"f_comp_subj=True", "f_com_obj=True", "ana_f_type=she", "ana_f_type=they", "ana_f_type=it", "ant_ner=DATE", "ant_rel=nmod:to", 
			"ana_rel=nmod:to", "ana_f_type=i", "ana_f_type=he", "head_contained=True", "ana_ner=GPE", "ant_type=LIST", "ant_ner=PRODUCT", 
			"acronym=True", "first_compatible_head_match=True", "ana_rel=iobj", "ant_ner=CARDINAL", "ana_rel=nmod:npmod", "ant_rel=nsubj:xsubj", 
			"ana_rel=nmod:in", "ana_type=PROPER", "ant_f_type=NAM", "ant_ner=O", "ana_ner=O", "ana_ner=PRODUCT", "ana_len=1", "ana_type=NOMINAL", 
			 "ana_type=PRONOMINAL", "ant_ner=PERSON", "ana_ner=PERSON", "ant_len=>5", "ana_len=3", "ant_f_type=INDEF", "ant_rel=nsubj", "ant_ner=ORG", 
			 "f_comp_att=True", "ant_rel=compound", "ant_ner=GPE", "ana_rel=no-parent", "ana_ner=DATE", "ant_f_type=Other", "ana_ner=ORG", "comp_animacy=True", 
			 "ant_type=NOMINAL", "comp_number=True", "comp_gender=True", "ant_len=1", "ana_rel=nsubj", "ant_type=PROPER", "ana_len=2", "ana_f_type=NAM", 
			 "ant_f_type=DEF", "ana_f_type=DEF");
	public List<String> importantAtts_train_pronoun_20_6_60= Arrays.asList("ana_type=PRONOMINAL", "ana_f_type=they", "ana_rel=nsubj:xsubj", "ant_rel=nsubj:xsubj", 
			"ana_rel=nmod:npmod", "f_com_obj=True", "ant_rel=iobj", "ant_ner=CARDINAL", "ant_ner=NORP", "f_comp_att=True", "ant_f_type=they", "ant_f_type=Other", 
			"ant_f_type=he", "ant_rel=nmod:poss", "ant_f_type=i", "ant_f_type=you", "ant_len=3", "ana_f_type=we", "ant_len=<=5", "ana_rel=dobj", "ana_f_type=it", 
			"head_contained=True", "ant_len=>5", "token_contained=True", "ant_f_type=she", "f_comp_subj=True", "ana_rel=iobj", "ana_f_type=she", "ant_f_type=it", 
			"ana_f_type=Other", "ana_rel=nsubjpass", "ant_f_type=we", "ant_rel=nsubjpass", "comp_animacy=True", "comp_number=True", "ant_rel=nsubj", "ana_rel=nsubj", 
			"ant_type=NOMINAL", "ana_len=1", "ana_ner=O", "ant_ner=PERSON", "ant_type=LIST", "ant_ner=ORG", "ant_ner=GPE", "ant_rel=compound", "ana_rel=nmod:poss", 
			"ant_ner=O", "comp_gender=True", "comp_attributes=True", "ant_len=1", "ant_f_type=INDEF", "ant_f_type=DEF", "ant_len=2", "ant_f_type=NAM", "ant_type=PROPER", 
			"ana_f_type=i", "ant_rel=dobj", "ana_f_type=you", "ana_f_type=he", "ant_type=PRONOMINAL");
	
	public List<String> importantAtts_train_nominal_20_6_60= Arrays.asList("ana_type=NOMINAL", "ana_f_type=DEF", "ana_rel=nmod:poss", "ant_f_type=INDEF", "ant_len=>5", 
			"ana_rel=nmod:from", "ant_rel=conj:and", "head_match=True", "token_contained=True", "ant_len=2", "ant_f_type=DEF", "ana_rel=dobj", "ant_type=PROPER", "ant_f_type=NAM", 
			"ana_len=3", "ant_rel=dobj", "ana_rel=nsubj", "ana_len=<=5", "ant_len=3", "ant_len=<=5", "ana_rel=nmod:of", "ant_rel=nmod:of", "ant_rel=nmod:poss", "ant_rel=nmod:in", 
			"ana_rel=nmod:in", "head_contained=True", "ana_rel=nsubjpass", "ant_rel=nmod:on", "ant_rel=nmod:from", "ant_rel=nmod:with", "ant_rel=nsubjpass", "ana_rel=nmod:on", 
			"ana_rel=nmod:with", "ant_rel=nmod:to", "ant_ner=ORG", "ant_ner=GPE", "ana_rel=nmod:to", "f_comp_att=True", "first_compatible_head_match=True", 
			"ant_rel=nmod:into", "ant_ner=EVENT", "f_com_obj=True", "f_comp_subj=True", "comp_attributes=True", "ana_len=2", "comp_number=True", 
			"ant_rel=nsubj", "comp_gender=True", "comp_animacy=True", "ant_ner=O", "ant_type=NOMINAL");
	public List<String> importantAtts_train_proper_20_6_60= Arrays.asList("ana_type=PROPER", "ant_len=<=5", "comp_gender=True", "ana_rel=nmod:poss", "ant_ner=GPE", "ant_rel=nmod:poss", "head_match=True", "f_comp_subj=True", "ant_rel=nmod:from", "ana_rel=nmod:with", "ana_rel=nmod:from", "ana_rel=nmod:tmod", "ana_rel=nmod:on", "ant_rel=appos", "ant_rel=nmod:for", "ant_rel=nmod:with", "ana_rel=nmod:to", "ant_rel=nmod:to", "ana_rel=appos", "ana_rel=no-parent", "head_contained=True", "token_contained=True", "ant_ner=ORG", "ant_rel=nmod:in", "ant_ner=DATE", "ant_ner=PRODUCT", "ana_rel=nmod:about", "first_compatible_head_match=True", "ana_rel=nmod:between", "ant_rel=nmod:about", "ant_ner=LOC", "f_com_obj=True", "ant_rel=dobj", "ana_ner=ORG", "ana_rel=nmod:in", "ana_rel=compound", "ant_rel=nmod:of" , "ana_rel=dobj", "ant_len=3", "ana_len=<=5", "ana_ner=GPE", "ant_ner=PERSON", "ana_ner=DATE", "ana_len=3", "ant_len=2", "ana_len=2", "ant_f_type=NAM", "ant_rel=nsubj", "comp_attributes=True", "ant_len=1", "ana_ner=O", "ant_type=PROPER", "ana_rel=nsubj", "ant_len=>5", "ana_ner=PERSON", "ana_len=1", "compatible_modifiers=True", "comp_animacy=True", "ant_ner=O", "comp_number=True", "ana_ner=PRODUCT", "f_comp_att=True", "ant_rel=conj:and", "ant_rel=compound", "ana_ner=LOC", "ana_rel=nmod:for", "ana_f_type=NAM", "acronym=True");
	
	public List<String> train_proper_pos_20_5_60 = Arrays.asList("ana_rel=nmod:poss", "ana_f_POS=NNP", "ant_head_POS=NNP", "ana_n_POS=NNS", "ana_p_POS=VBP", 
			"ana_p_POS=JJ", "ana_n_POS=NNP", "ana_p_POS=VBG", "ant_p_POS=DT", "ant_n_POS=TO", "ana_rel=nsubjpass", "ana_n_POS=TO", "ant_n_POS=JJ", "ana_n_POS=JJ", "ant_n_POS=NNS", 
			"ant_f_type=you", "ana_rel=nmod:for", "ana_p_POS=VB", "ana_n_POS=RB", "ant_rel=compound", "ant_n_POS=RB", "ana_n_POS=MD", "ant_rel=nmod:with", "ant_ner=PRODUCT", 
			"ana_p_POS=PRP", "first_compatible_head_match=True", "f_comp_subj=True", "ant_n_POS=NNP", "ana_ner=EVENT", "f_com_obj=True", "ant_l_POS=,", "f_comp_att=True", 
			"ana_p_POS=VBD", "ant_n_POS=MD", "ant_p_POS=VBZ", "ana_rel=nmod:to", "ant_f_type=i", "ana_rel=no-parent", "ant_p_POS=VBG", "ant_rel=nmod:to", "head_contained=True", 
			"token_contained=True", "ant_f_type=he", "ant_head_POS=PRP$", "ant_p_POS=CC", "ana_p_POS=CC", "ant_n_POS=CC", "ant_l_POS=PRP$", "ana_f_POS=JJ", "ant_f_POS=JJ", 
			"ana_n_POS=IN", "ant_rel=nmod:in", "ant_p_POS=VBD", "ant_n_POS=NN", "ana_p_POS=DT", "ana_head_POS=NNS", "ant_ner=ORG", "ant_p_POS=VB", "ant_rel=nmod:poss", 
			"ana_rel=dobj", "ant_n_POS=IN", "ana_head_POS=NN", "ana_rel=compound", "ana_n_POS=VBZ", "ant_n_POS=VBZ", "ant_p_POS=,", "ant_rel=nmod:of", "ana_p_POS=,", 
			"ant_f_POS=PRP$", "ana_n_POS=NN", "ana_ner=O", "ana_l_POS=NN", "ana_n_POS=,", "ant_n_POS=.", "ant_n_POS=,", "ana_p_POS=START", "ana_n_POS=.", "ana_ner=GPE", 
			"ana_n_POS=VBD", "ana_rel=nmod:in", "ant_n_POS=VBD", "ant_f_POS=PRP", "ant_rel=dobj", "ana_f_POS=DT", "ant_f_POS=NNP", "ant_type=PROPER", "comp_attributes=True", 
			"ant_l_POS=NNP", "ant_l_POS=NN", "ana_rel=nsubj", "comp_number=True", "acronym=True", "ant_ner=EVENT", "ana_ner=PRODUCT", "head_match=True", "ant_rel=conj:and", 
			"ant_l_POS=POS", "ant_ner=GPE", "ana_rel=appos", "ant_ner=DATE", "ana_l_POS=POS", "ana_ner=ORG", "ant_type=PRONOMINAL", "ant_head_POS=NNS", "ant_p_POS=START", 
			"ant_ner=PERSON", "ana_ner=DATE", "ant_head_POS=PRP", "ant_l_POS=PRP", "ana_head_POS=NNP", "ant_ner=O", "comp_animacy=True", "ant_f_type=NAM", "ana_ner=PERSON", 
			"ant_rel=nsubj", "ant_head_POS=NN", "ant_f_POS=DT", "ana_p_POS=IN", "ant_p_POS=IN", "comp_gender=True", "ana_l_POS=NNP");
	public List<String> train_nominal_pos_20_5_60 = Arrays.asList("ana_f_POS=DT", "token_contained=True", "f_comp_att=True", 
			"first_compatible_head_match=True", "ana_rel=det", "ant_ner=EVENT", "ana_n_POS=NN", "f_com_obj=True", "ana_n_POS=JJ", "ana_l_POS=DT", "ant_f_type=they", 
			"ana_head_POS=DT", "ant_ner=ORG", "ant_p_POS=VBD", "ant_n_POS=NN", "ant_ner=GPE", "ana_f_type=INDEF", "ant_type=NOMINAL", "ana_f_type=DEF", "ant_f_POS=DT", 
			"ant_f_type=DEF", "ana_p_POS=START", "ant_rel=dobj", "ant_p_POS=START", "ant_type=PROPER", "ant_f_type=NAM", "comp_attributes=True", "ana_l_POS=NN", "ana_p_POS=IN", 
			"ant_p_POS=IN", "ana_head_POS=NN", "comp_animacy=True", "comp_number=True", "ant_ner=O", "ant_rel=nsubj", "ant_l_POS=NN", "ant_f_type=INDEF", "ana_rel=nsubj", 
			"ant_l_POS=NNP", "ant_head_POS=NNP", "ant_f_POS=PRP$", "ant_f_POS=NNP", "ant_l_POS=POS", "ant_head_POS=NN", "comp_gender=True", "ant_l_POS=DT", "ana_l_POS=POS", 
			"ana_rel=nmod:poss", "head_match=True", "ana_n_POS=VBD", "ant_n_POS=VBZ", "ant_p_POS=,", "ant_p_POS=VB", "ant_rel=nmod:poss", "ant_rel=nmod:in", "ant_n_POS=IN", 
			"ana_n_POS=VBZ", "ant_n_POS=VBD", "head_contained=True", "f_comp_subj=True", "ana_n_POS=RB");
	public List<String> train_pronoun_pos_20_5_60 = Arrays.asList("ana_f_type=i", "ant_f_type=i", "ana_rel=nsubj:xsubj", "ana_p_POS=WDT", "ant_p_POS=``", 
			"ant_head_POS=NNPS", "ana_n_POS=NNP", "ant_p_POS=NNS", "ant_rel=iobj", "ana_p_POS=''", "ana_p_POS=NNP", "ant_ner=CARDINAL", "ant_l_POS=NNPS", "ant_ner=NORP", 
			"f_com_obj=True", "ana_p_POS=``", "f_comp_att=True", "ana_f_type=they", "ana_f_type=you", "ana_n_POS=VBD", "ant_l_POS=NNP", "ant_rel=dobj", "ana_f_POS=PRP$", 
			"ana_l_POS=PRP$", "ana_p_POS=VBD", "ana_n_POS=NNS", "ant_n_POS=NN", "ant_f_type=Other", "ant_l_POS=PRP$", "ant_head_POS=PRP$", "ant_n_POS=MD", "ant_f_type=they", 
			"ana_f_type=she", "ant_rel=nmod:poss", "ana_n_POS=.", "ant_f_type=you", "ana_p_POS=,", "ant_f_type=he", "ant_p_POS=VBD", "ant_n_POS=IN", "ant_f_POS=JJR", 
			"ana_p_POS=RP", "ana_rel=nmod:npmod", "ant_rel=nsubj:xsubj", "ant_p_POS=WDT", "f_comp_subj=True", "ana_n_POS=TO", "ant_rel=nmod:to", "ant_rel=nsubjpass", 
			"ant_f_POS=CD", "ana_p_POS=VBZ", "ana_n_POS=,", "ant_p_POS=VBG", "ant_p_POS=NN", "ana_p_POS=WRB", "ana_p_POS=VBG", "ana_p_POS=RB", "ana_p_POS=VBN", 
			"ant_head_POS=JJ", "ant_p_POS=WRB", "ant_n_POS=TO", "ana_rel=iobj", "ant_n_POS=VB", "ant_l_POS=JJ", "ana_p_POS=WP", "ant_f_type=she", "ant_n_POS=JJ", 
			"ant_n_POS=NNS", "ana_rel=nsubjpass", "ant_n_POS=RB", "ant_f_POS=JJ", "ant_head_POS=DT", "ana_p_POS=NN", "ant_n_POS=CC", "ana_p_POS=VBP", "ana_p_POS=CC", 
			"ana_n_POS=JJ", "ant_f_type=it", "ant_p_POS=VBZ", "ant_f_type=we", "ant_p_POS=VBP", "ant_p_POS=CC", "ant_f_POS=NNS", "ant_f_POS=NN", "ana_n_POS=IN", 
			"ana_f_type=Other", "ana_n_POS=RB", "ant_f_POS=JJS", "compatible_modifiers=True", "ana_type=PRONOMINAL", "ana_ner=O", "comp_animacy=True", 
			"comp_attributes=True", "ana_p_POS=START", "ant_p_POS=IN", "ant_f_POS=DT", "ant_type=PRONOMINAL", "ana_p_POS=IN", "ant_n_POS=.", "ant_f_type=DEF", 
			"ant_f_type=INDEF", "ant_head_POS=PRP", "ant_f_type=NAM", "ana_f_type=he", "ana_n_POS=VBP", "ant_type=PROPER", "ana_f_POS=PRP", "ana_head_POS=PRP", 
			"ant_ner=O", "ana_l_POS=PRP", "ant_n_POS=VBP", "ana_n_POS=NN", "ant_n_POS=VBZ", "ana_p_POS=VB", "ant_p_POS=VB", "ana_n_POS=MD", "ana_f_type=we", 
			"ant_n_POS=VBD", "ana_rel=dobj", "ant_n_POS=,", "head_contained=True", "token_contained=True", "ana_f_type=it", "ana_n_POS=VBZ", "ant_head_POS=NNS", 
			"comp_number=True", "comp_gender=True", "f_com_obj=False", "ana_rel=nsubj", "ant_head_POS=NNP", "ana_rel=nmod:poss", "ana_head_POS=PRP$", "ant_f_POS=NNP", 
			"ant_l_POS=NNS", "ant_ner=PERSON", "ant_p_POS=,", "ant_rel=compound", "ana_p_POS=:", "ant_f_POS=PDT", "ant_l_POS=POS", "ant_ner=ORG", "ant_type=LIST", 
			"ant_ner=GPE", "ant_f_POS=PRP$", "ant_rel=nsubj", "ant_l_POS=PRP", "ant_head_POS=NN", "ant_l_POS=NN", "ant_f_POS=PRP", "ant_p_POS=START", "ant_type=NOMINAL");
	public List<String> train_list_pos_20_5_60 = Arrays.asList("ana_rel=nsubj", "ant_f_type=you", "ant_f_POS=PRP", "ana_n_POS=,", "ana_head_POS=NNS", 
			"ana_l_POS=NNS", "ana_ner=ORG", "ana_f_POS=NNS", "ana_p_POS=START", "ant_l_POS=PRP", "comp_animacy=False", "token_contained=False", "ant_head_POS=PRP", 
			"comp_attributes=False");
	
	public List<String> importantAtts;
	public List<String> common_atts;
	public List<String> proper_specific_atts;
	public List<String> nominal_specific_atts;
	public List<String> pronoun_specific_atts;
	public List<String> list_specific_atts;
	public int attribute_size = 0;
	
	public CategoricalFeatureExtractor(Properties props, Dictionaries dictionaries) {
		this.dictionaries = dictionaries;
		conll = CorefProperties.conll(props);
		parser = new FeatureFileParser();
		counts = new int[parser.m_allFeatures.size()];
		Set<String> union = new HashSet<String>();
		union.addAll(train_proper_pos_20_5_60);
		union.addAll(train_nominal_pos_20_5_60);
		union.addAll(train_pronoun_pos_20_5_60);
		
		common_atts = new ArrayList<String>();
		common_atts.addAll(union);
		
//		common_atts.addAll(train_proper_pos_20_5_60);
//		common_atts.retainAll(train_nominal_pos_20_5_60);
//		common_atts.retainAll(train_pronoun_pos_20_5_60);
		proper_specific_atts = new ArrayList<String>();
//		proper_specific_atts.addAll(train_proper_pos_20_5_60);
//		proper_specific_atts.removeAll(common_atts);
		nominal_specific_atts = new ArrayList<String>();
//		nominal_specific_atts.addAll(train_nominal_pos_20_5_60);
//		nominal_specific_atts.removeAll(common_atts);
		pronoun_specific_atts = new ArrayList<String>();
//		pronoun_specific_atts.addAll(train_pronoun_pos_20_5_60);
//		pronoun_specific_atts.removeAll(common_atts);
		list_specific_atts = new ArrayList<String>();
//		list_specific_atts.addAll(train_list_pos_20_5_60);
//		list_specific_atts.removeAll(common_atts);
		attribute_size = common_atts.size() + proper_specific_atts.size() + nominal_specific_atts.size()+pronoun_specific_atts.size()+list_specific_atts.size();
		
		if (conll) {
			genres = new HashMap<>();
			genres.put("bc", 0);
			genres.put("bn", 1);
			genres.put("mz", 2);
			genres.put("nw", 3);
			boolean english = CorefProperties.getLanguage(props) == Locale.ENGLISH;
			if (english) {
				genres.put("pt", 4);
			}
			genres.put("tc", english ? 5 : 4);
			genres.put("wb", english ? 6 : 5);
		} else {
			genres = null;
		}
	}


	public SimpleMatrix getPairFeatures(Pair<Integer, Integer> pair, Document document,
			Map<Integer, List<Mention>> mentionsByHeadIndex) {
		Mention m1 = document.predictedMentionsByID.get(pair.first);
		Mention m2 = document.predictedMentionsByID.get(pair.second);
		List<Integer> featureVals = pairwiseFeatures(document, m1, m2, dictionaries, conll, false);
		SimpleMatrix features = new SimpleMatrix(featureVals.size(), 1);
		for (int i = 0; i < featureVals.size(); i++) {
			features.set(i, featureVals.get(i));
		}
		features = NeuralUtils.concatenate(features,
				encodeDistance(m2.sentNum - m1.sentNum),
				encodeDistance(m2.mentionNum - m1.mentionNum - 1),
				new SimpleMatrix(new double[][] {{
					m1.sentNum == m2.sentNum && m1.endIndex > m2.startIndex ? 1 : 0}}),
					getMentionFeatures(m1, document, mentionsByHeadIndex),
					getMentionFeatures(m2, document, mentionsByHeadIndex),
					encodeGenre(document));

		return features;
	}

	public static boolean[] anaphoricityFeatures(Document document, Mention m1, List<Mention> antecedents,
			Dictionaries dictionaries) {
		boolean[] features = new boolean[11];

		for (int i = 0; i < features.length; i++)
			features[i] = false;

		for (Mention ante : antecedents){
			boolean earlier = ante.appearEarlierThan(m1);
			if (ante != m1 && m1.insideIn(ante))
				features[10] = true;

			if (ante != m1 && !(m1.insideIn(ante) || ante.insideIn(m1))){
				if (m1.toString().trim().equalsIgnoreCase(ante.toString().trim())){
					features[0] = true;
					if (earlier)
						features[1] = true;
				}

				if (m1.headString.equalsIgnoreCase(ante.headString)){
					features[2] = true;
					if (earlier)
						features[3] = true;
				}

				//					if (m1.headsAgree(ante)){
				//						features[4] = true;
				//						if (earlier)
				//							features[5] = true;
				//					}

				if (m1.headString.toLowerCase().contains(ante.headString.toLowerCase()) || ante.headString.toLowerCase().contains(m1.headString.toLowerCase())){
					features[4] = true;
					if (earlier)
						features[5] = true;
				}

				if (m1.spanToString().toLowerCase().contains(ante.spanToString().toLowerCase())){
					features[6] = true;
					if (earlier)
						features[7]  = true;

				}
				if (ante.spanToString().toLowerCase().contains(m1.spanToString().toLowerCase())){
					features[8] = true;
					if (earlier)
						features[9]  = true;

				}
				//					if(edu.stanford.nlp.coref.statistical.FeatureExtractor.relaxedStringMatch(m1, ante)){
				//						features[10] = true;
				//						if(earlier)
				//							features[11] = true;
				//					}
			}


		}

		return features;
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
	public static List<Integer> pairwiseFeatures(Document document, Mention m1, Mention m2,
			Dictionaries dictionaries, boolean isConll, boolean add_nam_features) {
		String speaker1 = m1.headWord.get(CoreAnnotations.SpeakerAnnotation.class);
		String speaker2 = m2.headWord.get(CoreAnnotations.SpeakerAnnotation.class);
		List<Integer> features = new ArrayList<>();
		features.add(isConll ? (speaker1.equals(speaker2) ? 1 : 0) : 0);
		features.add(isConll ?
				(CorefRules.antecedentIsMentionSpeaker(document, m2, m1, dictionaries) ? 1 : 0) : 0);
		features.add(isConll ?
				(CorefRules.antecedentIsMentionSpeaker(document, m1, m2, dictionaries) ? 1 : 0) : 0);
		features.add(m1.headsAgree(m2) ? 1 : 0);
		features.add(
				m1.toString().trim().toLowerCase().equals(m2.toString().trim().toLowerCase()) ? 1 : 0);
		features.add(edu.stanford.nlp.coref.statistical.FeatureExtractor.relaxedStringMatch(m1, m2)
				? 1 : 0);
		//		if (add_nam_features){
		//			String genre = document.docInfo.get("DOC_ID").split("/")[0];
		//
		//			boolean[] nam_features = NAM_minedFeatures_20_4_40(genre, m1, m2, speaker1.equals(speaker2));
		//			for (boolean v : nam_features)
		//				features.add(v ? 1 : 0);
		//		}
		return features;
	}

	public static List<Integer> pairwiseFeaturesPlus(Document document, Mention m1, Mention m2,
			Dictionaries dictionaries, boolean isConll, boolean add_nam_features) {
		String speaker1 = m1.headWord.get(CoreAnnotations.SpeakerAnnotation.class);
		String speaker2 = m2.headWord.get(CoreAnnotations.SpeakerAnnotation.class);
		List<Integer> features = new ArrayList<>();
		features.add(isConll ? (speaker1.equals(speaker2) ? 1 : 0) : 0);
		features.add(isConll ?
				(CorefRules.antecedentIsMentionSpeaker(document, m2, m1, dictionaries) ? 1 : 0) : 0);
		features.add(isConll ?
				(CorefRules.antecedentIsMentionSpeaker(document, m1, m2, dictionaries) ? 1 : 0) : 0);
		features.add(m1.headsAgree(m2) ? 1 : 0);
		boolean exactMatch = m1.toString().trim().toLowerCase().equals(m2.toString().trim().toLowerCase()); 
		features.add(exactMatch ? 1 : 0);
		features.add(edu.stanford.nlp.coref.statistical.FeatureExtractor.relaxedStringMatch(m1, m2)
				? 1 : 0);
		boolean headMatch = m1.headString.equalsIgnoreCase(m2.headString); 
		features.add(!exactMatch && headMatch ? 1 : 0);
		features.add(!exactMatch && !headMatch && m2.toString().trim().toLowerCase().contains(m1.toString().trim().toLowerCase()) ? 1 : 0);
		features.add(!exactMatch && !headMatch && m1.toString().trim().toLowerCase().contains(m2.toString().trim().toLowerCase()) ? 1 : 0);
		features.add(!exactMatch && !headMatch && m2.toString().trim().toLowerCase().contains(m1.headString.toLowerCase()) ? 1 : 0);
		features.add(!exactMatch && !headMatch && m1.toString().trim().toLowerCase().contains(m2.headString.toLowerCase()) ? 1 : 0);
		return features;
	}
	//	static boolean[] NAM_minedAttributes_10_5_50( String genre, Mention ant, Mention ana, boolean sameSpeaker){
	//		boolean[] all_useful_att = new boolean[namFeatureNum];
	//		if (ana.mentionType != MentionType.PROPER)
	//			return all_useful_att;
	//
	//		boolean exact_match = ant.toString().trim().toLowerCase().equals(ana.toString().trim().toLowerCase());
	//		boolean head_match = ant.headString.equalsIgnoreCase(ana.headString);
	//		boolean tokens_contained = ant.toString().trim().toLowerCase().contains(ana.toString().trim().toLowerCase())
	//				|| ana.toString().trim().toLowerCase().contains(ant.toString().trim().toLowerCase());
	//		boolean head_contained = ant.headString.toLowerCase().contains(ana.headString.toLowerCase())
	//				|| ana.headString.toLowerCase().contains(ant.headString.toLowerCase());
	//
	//		String ana_deprel = get_deprel(ana);
	//		String ant_deprel = get_deprel(ant);
	//		int ana_length = ana.endIndex - ana.startIndex;
	//		int ant_length = ant.endIndex - ant.startIndex;
	//		boolean ant_is_DEM = ant.sentenceWords.get(ant.headIndex).get(CoreAnnotations.PartOfSpeechAnnotation.class).startsWith("DT");


	//		if (head_match)
	//			all_useful_att[] = true;
	//		if (head_contained)
	//			all_useful_att[] = true;
	//		if (tokens_contained)
	//			all_useful_att[] = true;
	//		if (ana_length == 1)
	//			all_useful_att[] = true;
	//		if (ant_length == 1)
	//			all_useful_att[] = true;
	//		if (ana_deprel.equals("nmod:tmod"))
	//			all_useful_att[] = true;
	//		if (ana_deprel.equals("nmod:poss"))
	//			all_useful_att[] = true;
	//		if (ana_deprel.equals("compound"))
	//			all_useful_att[] = true;
	//		if (ana_deprel.equals("nmod"))
	//			all_useful_att[] = true;
	//		if (ant.sentNum==ana.sentNum)
	//			all_useful_att[] = true;
	//		if (ana.sentNum-ant.sentNum == 1)
	//			all_useful_att[] = true;
	//		if (ana.sentNum-ant.sentNum == 2)
	//			all_useful_att[] = true;
	//		if(ant_deprel.equals("compound"))
	//			all_useful_att[] = true;
	//		if(ant_deprel.equals("dobj"))
	//			all_useful_att[] = true;
	//		if (ant_deprel.equals("det"))
	//			all_useful_att[] = true;
	//		if (ant_deprel.equals("nmod"))
	//			all_useful_att[] = true;
	//		if (ana.nerString.equals("NUMBER"))
	//			all_useful_att[] = true;
	//		if (ana.nerString.equals("DATE"))
	//			all_useful_att[] = true;
	//		if (ana.nerString.startsWith("ORG"))
	//			all_useful_att[] = true;
	//		if (ana.nerString.startsWith("PER"))
	//			all_useful_att[] = true;
	//		if (ana.nerString.startsWith("GPE"))
	//			all_useful_att[] = true;
	//		if (ant.nerString.equals("NUMBER"))
	//			all_useful_att[] = true;
	//		if (ant.nerString.equals("DATE"))
	//			all_useful_att[] = true;
	//		if (ant.nerString.startsWith("GPE"))
	//			all_useful_att[] = true;
	//		if (ant.nerString.startsWith("ORG"))
	//			all_useful_att[] = true;
	//		if (ant.nerString.startsWith("PER"))
	//			all_useful_att[] = true;
	//		if (ant_is_DEM)
	//			all_useful_att[] = true;
	//		if (ant.mentionType == MentionType.PROPER)
	//			all_useful_att[] = true;
	//		if(ant.animacy == Animacy.ANIMATE)
	//			all_useful_att[] = true;
	//		if(ant.number == Number.PLURAL)
	//			all_useful_att[] = true;


	//		return all_useful_att;

	/*
  ant_sem_class=OBJECT , ana_number=SINGULAR , same_speaker=True , embedding=False , compatible_gender=True , ant_length=>5 , head_match=True , ant_deprel=nmod:tmod , alias=False , exact_match=False , ana_fine_type=NAM , ana_deprel=appos , ana_gender=MALE , ant_deprel=nmod:poss , ant_head_ner=PERSON , ana_deprel=nsubj , ant_number=PLURAL , ant_deprel=nsubj , compatible_number=False , modifier=False , token_distance=>=10 , ana_gender=NEUTRAL , same_gender=False , genre=nw , same_number=True , same_gender=True , compatible_gender=False , ana_sem_class=OBJECT , ant_gender=NEUTRAL , sentence_distance=>=5 , ant_number=SINGULAR , compatible_number=True ,

	 */

	//	}

	//	static boolean[] NAM_minedFeatures_20_4_40( String genre, Mention ant, Mention ana, boolean sameSpeaker){
	//		boolean[] all_features = new boolean[namFeatureNum];
	//		if (ana.mentionType != MentionType.PROPER)
	//			return all_features;
	//		
	//		boolean exact_match = ant.toString().trim().toLowerCase().equals(ana.toString().trim().toLowerCase());
	//		boolean head_match = ant.headString.equalsIgnoreCase(ana.headString);
	//		boolean tokens_contained = ant.toString().trim().toLowerCase().contains(ana.toString().trim().toLowerCase())
	//				|| ana.toString().trim().toLowerCase().contains(ant.toString().trim().toLowerCase());
	//		boolean head_contained = ant.headString.toLowerCase().contains(ana.headString.toLowerCase())
	//				|| ana.headString.toLowerCase().contains(ant.headString.toLowerCase());
	//
	//		String ana_deprel = get_deprel(ana);
	//		String ant_deprel = get_deprel(ant);
	//		int ana_length = ana.endIndex - ana.startIndex;
	//		int ant_length = ant.endIndex - ant.startIndex;
	//
	//		if (exact_match){
	//			if (ant.nerString.equals("NUMBER")){
	//				all_features[0] = true;
	//			}
	//			if (ant.nerString.equals("DATE")){
	//				all_features[1] = true;
	//			}
	//			if (ana_deprel.equals("nmod:poss")){
	//				all_features[2] = true;
	//			}
	//			if (ana.nerString.equals("DATE"))
	//				all_features[3] = true;
	//			if (ana.nerString.startsWith("ORG") && ant_deprel.equals("nsubj"))
	//				all_features[4] = true;
	//			if (ana.nerString.equals("NUMBER"))
	//				all_features[5] = true;
	//			if (ana.sentNum-ant.sentNum <=1)
	//				all_features[6] = true;
	//			if (ant_deprel.equals("nmod:poss"))
	//				all_features[7] = true;
	//			if (ant_deprel.equals("nsubj"))
	//				all_features[8] = true;
	//		}
	//		if (head_match){
	//			if(ant.nerString.equals("NUMBER") && (ana_length == 1 || ant_length == 1))
	//				all_features[9]=true;
	//			if(ant.nerString.equals("DATE") && (ana_length == 1 || ant_length == 1))
	//				all_features[10]=true;
	//			if(ana_deprel.equals("nmod:tmod") && ana_length == 1)
	//				all_features[11]=true;
	//			if(ana_deprel.equals("nmod:tmod") && tokens_contained)
	//				all_features[12]=true;
	//			if(ana_deprel.equals("nmod:poss") && ana_length == 1)
	//				all_features[13]=true;
	//			if(ana.number == Number.SINGULAR)
	//				all_features[14]=true;
	//			if(ant.gender == Gender.NEUTRAL)
	//				all_features[15]=true;
	//			if(ana.compatibleModifiers(ant))
	//				all_features[16]=true;
	//			if(ana.gender == Gender.NEUTRAL)
	//				all_features[17]=true;
	//			if(ant.number == Number.SINGULAR)
	//				all_features[18]=true;
	//			if(ant.mentionType == MentionType.PROPER)
	//				all_features[19]=true;
	//			if(ana.gendersAgree(ant))
	//				all_features[20]=true;
	//			if(tokens_contained)
	//				all_features[21]=true;
	//			if(ana.nerString.equals("DATE") && (ana_length == 1 || ant_length == 1))
	//				all_features[22]=true;
	//			if(ant_deprel.equals("nmod:tmod")&& ant_length == 1)
	//				all_features[23]=true;
	//			if(ant_deprel.equals("nmod:tmod") && ana.compatibleModifiers(ant))
	//				all_features[24]=true;
	//			if(ant_deprel.equals("nmod:tmod") && tokens_contained)
	//				all_features[25]=true;
	//			if(ant_deprel.equals("nmod:poss") && ant_length == 1)
	//				all_features[26]=true;
	//			if(ana.nerString.equals("NUMBER") && (ana_length == 1 || ant_length == 1))
	//				all_features[27]=true;
	//			if(ana.sentNum-ant.sentNum <= 1 && ana_length == 1)
	//				all_features[28]=true;
	//			if(ana.sentNum-ant.sentNum <= 1 && ant_length == 1)
	//				all_features[29]=true;
	//			if(ana.sentNum-ant.sentNum <= 1 && genre.equals("wb"))
	//				all_features[30]=true;
	//		}
	//		if (head_contained){
	//			if(ant.nerString.equals("DATE") && ana_length == 1)
	//				all_features[31]=true;
	//			if(ana_deprel.equals("nmod:tmod") && ana_length == 1)
	//				all_features[32]=true;
	//			if(ana_deprel.equals("nmod:poss") && ana_length == 1)
	//				all_features[33]=true;
	//			if(ant_deprel.equals("nmod:tmod") && tokens_contained)
	//				all_features[34]=true;
	//		}
	//		if (tokens_contained){
	//			if(ant.nerString.equals("NUMBER") && ant_length == 1)
	//				all_features[35]=true;
	//			if(ant_deprel.equals("nmod:tmod") && ant_length == 1)
	//				all_features[36]=true;
	//			if(ant_deprel.equals("nmod:tmod") && ana.nerString.equals("DATE"))
	//				all_features[37]=true;
	//			if(ant_deprel.equals("nmod:tmod") && ana.compatibleModifiers(ant))
	//				all_features[38]=true;
	//		}
	//		return all_features;
	//	}

	static String get_deprel(Mention m){
		Iterator<SemanticGraphEdge> iterator =
				m.enhancedDependency.incomingEdgeIterator(m.headIndexedWord);
		SemanticGraphEdge relation = iterator.hasNext() ? iterator.next() : null;
		String depRelation = relation == null ? "no-parent" : relation.getRelation().toString();
		return depRelation;
	}
	public Map<Integer, Map<Integer, List<Integer>>> extendedPairwiseFeatures(Document document, List<Mention> mentionList,
			Dictionaries dictionaries, boolean isConll) {

		Map<Integer, Map<Integer, List<Integer>>> allFeatures = new HashMap<Integer, Map<Integer,List<Integer>>>();
		for (int i = 1; i < mentionList.size(); i++){
			Mention m1 = mentionList.get(i);
			Set<String> m1AttVals = parser.getMentionFeatures(m1, "ana");

			List<Mention> allAntecedents = mentionList.subList(0, i);
			Map<Integer, List<Integer>> anaFeatures = new HashMap<Integer, List<Integer>>();

			List<Set<String>> allPairwise = parser.getPairwiseFeatures(document, m1, allAntecedents, dictionaries);
			for (int k = allAntecedents.size()-1; k >= 0; k--){
				List<Integer> features = new ArrayList<Integer>();
				Mention m2 = allAntecedents.get(k);
				Set<String> m2AttVals = parser.getMentionFeatures(m2, "ant");
				Set<String> pairAttVals = allPairwise.get(allAntecedents.size()-1-k);

				String speaker1 = m1.headWord.get(CoreAnnotations.SpeakerAnnotation.class);
				String speaker2 = m2.headWord.get(CoreAnnotations.SpeakerAnnotation.class);
				features.add(isConll ? (speaker1.equals(speaker2) ? 1 : 0) : 0);
				features.add(isConll ?
						(CorefRules.antecedentIsMentionSpeaker(document, m2, m1, dictionaries) ? 1 : 0) : 0);
				features.add(isConll ?
						(CorefRules.antecedentIsMentionSpeaker(document, m1, m2, dictionaries) ? 1 : 0) : 0);
				features.add(m1.headsAgree(m2) ? 1 : 0);
				features.add(
						m1.toString().trim().toLowerCase().equals(m2.toString().trim().toLowerCase()) ? 1 : 0);
				features.add(edu.stanford.nlp.coref.statistical.FeatureExtractor.relaxedStringMatch(m1, m2)
						? 1 : 0);

				boolean[] newFeatures = parser.getFeatures(m1AttVals, m2AttVals, pairAttVals);
				for (int j = 0; j < newFeatures.length; j++){
					features.add(newFeatures[j] ? 1 : 0);
					if (newFeatures[j]){
						counts[j]++;
					}
				}


				anaFeatures.put(m2.mentionID, features);
			}
			allFeatures.put(m1.mentionID, anaFeatures);
		}
		return allFeatures;
	}

	public Map<Integer, Map<Integer, List<Integer>>> extendedPairwiseAttributes(Document document, List<Mention> mentionList,
			Dictionaries dictionaries, boolean isConll) {

		Map<Integer, Map<Integer, List<Integer>>> allFeatures = new HashMap<Integer, Map<Integer,List<Integer>>>();
		for (int i = 1; i < mentionList.size(); i++){
			Mention m1 = mentionList.get(i);
			Set<String> m1AttVals = parser.getMentionFeatures(m1, "ana");

			List<Mention> allAntecedents = mentionList.subList(0, i);
			Map<Integer, List<Integer>> anaFeatures = new HashMap<Integer, List<Integer>>();

			List<Set<String>> allPairwise = parser.getPairwiseFeatures(document, m1, allAntecedents, dictionaries);
			for (int k = allAntecedents.size()-1; k >= 0; k--){
				List<Integer> features = new ArrayList<Integer>();
				Mention m2 = allAntecedents.get(k);
				Set<String> m2AttVals = parser.getMentionFeatures(m2, "ant");
				Set<String> pairAttVals = allPairwise.get(allAntecedents.size()-1-k);

				String speaker1 = m1.headWord.get(CoreAnnotations.SpeakerAnnotation.class);
				String speaker2 = m2.headWord.get(CoreAnnotations.SpeakerAnnotation.class);
				features.add(isConll ? (speaker1.equals(speaker2) ? 1 : 0) : 0);
				features.add(isConll ?
						(CorefRules.antecedentIsMentionSpeaker(document, m2, m1, dictionaries) ? 1 : 0) : 0);
				features.add(isConll ?
						(CorefRules.antecedentIsMentionSpeaker(document, m1, m2, dictionaries) ? 1 : 0) : 0);
				features.add(m1.headsAgree(m2) ? 1 : 0);
				features.add(
						m1.toString().trim().toLowerCase().equals(m2.toString().trim().toLowerCase()) ? 1 : 0);
				features.add(edu.stanford.nlp.coref.statistical.FeatureExtractor.relaxedStringMatch(m1, m2)
						? 1 : 0);

				Set<String> allAttr = new HashSet<>();
				allAttr.addAll(m1AttVals);
				allAttr.addAll(m2AttVals);
				allAttr.addAll(pairAttVals);
				
				for (int jj = 0; jj < common_atts.size(); jj++){
					features.add(allAttr.contains(common_atts.get(jj)) ? 1 : 0);
				}
				for (int jj = 0; jj < proper_specific_atts.size(); jj++){
					if (m1.mentionType == MentionType.PROPER)
						features.add(allAttr.contains(proper_specific_atts.get(jj)) ? 1 : 0);
					else
						features.add(0);
				}
				for (int jj = 0; jj < nominal_specific_atts.size(); jj++){
					if (m1.mentionType == MentionType.NOMINAL)
						features.add(allAttr.contains(nominal_specific_atts.get(jj)) ? 1 : 0);
					else
						features.add(0);
				}
				for (int jj = 0; jj < pronoun_specific_atts.size(); jj++){
					if (m1.mentionType == MentionType.PRONOMINAL)
						features.add(allAttr.contains(pronoun_specific_atts.get(jj)) ? 1 : 0);
					else
						features.add(0);
				}
				for (int jj = 0; jj < list_specific_atts.size(); jj++){
					if (m1.mentionType == MentionType.LIST)
						features.add(allAttr.contains(list_specific_atts.get(jj)) ? 1 : 0);
					else
						features.add(0);
				}
				anaFeatures.put(m2.mentionID, features);
			}
			allFeatures.put(m1.mentionID, anaFeatures);
		}
		return allFeatures;
	}

	public Map<Integer, Map<Integer, List<Integer>>> selectedAntePairwiseFeatures(Document document, List<Mention> mentionList,
			Dictionaries dictionaries, boolean isConll) {


		Map<Integer, Map<Integer, List<Integer>>> allFeatures = new HashMap<Integer, Map<Integer,List<Integer>>>();
		for (int i = 1; i < mentionList.size(); i++){
			Mention m1 = mentionList.get(i);
			List<Mention> allAntecedents = mentionList.subList(0, i);

			ArrayList<MentionType> acceptedAntecedents = new ArrayList<MentionType>();

			if (m1.mentionType == MentionType.PROPER || m1.mentionType == MentionType.NOMINAL){
				boolean hasPrivAnte = false;
				for (Mention m : allAntecedents)
					if (m.mentionType == m1.mentionType || m.mentionType == MentionType.PROPER){
						hasPrivAnte = true;
						break;
					}
				if (hasPrivAnte && m1.mentionType == MentionType.PROPER)
					acceptedAntecedents.add(MentionType.PROPER);

				else if (hasPrivAnte && m1.mentionType == MentionType.NOMINAL)
					acceptedAntecedents.addAll(Arrays.asList(MentionType.PROPER, MentionType.NOMINAL));
				else
					acceptedAntecedents.addAll(Arrays.asList(MentionType.PROPER, MentionType.PRONOMINAL, MentionType.NOMINAL, MentionType.LIST));

			}

			Map<Integer, List<Integer>> anaFeatures = new HashMap<Integer, List<Integer>>();

			for (int k = allAntecedents.size()-1; k >= 0; k--){
				List<Integer> features = new ArrayList<Integer>();
				Mention m2 = allAntecedents.get(k);
				if (acceptedAntecedents.contains(m2.mentionType)){

					String speaker1 = m1.headWord.get(CoreAnnotations.SpeakerAnnotation.class);
					String speaker2 = m2.headWord.get(CoreAnnotations.SpeakerAnnotation.class);
					features.add(isConll ? (speaker1.equals(speaker2) ? 1 : 0) : 0);
					features.add(isConll ?
							(CorefRules.antecedentIsMentionSpeaker(document, m2, m1, dictionaries) ? 1 : 0) : 0);
					features.add(isConll ?
							(CorefRules.antecedentIsMentionSpeaker(document, m1, m2, dictionaries) ? 1 : 0) : 0);
					features.add(m1.headsAgree(m2) ? 1 : 0);
					features.add(
							m1.toString().trim().toLowerCase().equals(m2.toString().trim().toLowerCase()) ? 1 : 0);
					features.add(edu.stanford.nlp.coref.statistical.FeatureExtractor.relaxedStringMatch(m1, m2)
							? 1 : 0);
				}
				else
					features = null;

				anaFeatures.put(m2.mentionID, features);
			}
			allFeatures.put(m1.mentionID, anaFeatures);
		}
		return allFeatures;
	}

	//	private static boolean[] firstRichNominalExactMatch(Mention ana, List<Mention> antecedents){
	//		boolean[] ret = new boolean[antecedents.size()];
	//		int i = antecedents.size()-1;
	//		for (int k = 0; k < antecedents.size(); k++)
	//			ret[k] = false;
	//
	//		boolean found = false;
	//		while(i>=0 && !found){
	//			if (ana.nominalRichExactMatch(antecedents.get(i))){
	//				ret[i] = true;
	//				found = true;
	//			}
	//		}
	//		return ret;
	//	}
	//
	//	private static boolean[] firstRichNominalHeadMatch(Mention ana, List<Mention> antecedents){
	//		boolean[] ret = new boolean[antecedents.size()];
	//		int i = antecedents.size()-1;
	//		for (int k = 0; k < antecedents.size(); k++)
	//			ret[k] = false;
	//
	//		boolean found = false;
	//		while(i>=0 && !found){
	//			if (ana.nominalRichHeadMatch(antecedents.get(i))){
	//				ret[i] = true;
	//				found = true;
	//			}
	//		}
	//		return ret;
	//	}
	//
	//	private static boolean[] firstCompatibleHeadMatch(Mention ana, List<Mention> antecedents, Dictionaries dict){
	//		boolean[] ret = new boolean[antecedents.size()];
	//		int i = antecedents.size()-1;
	//		for (int k = 0; k < antecedents.size(); k++)
	//			ret[k] = false;
	//
	//		boolean found = false;
	//		while(i>=0 && !found){
	//			if (ana.compatibleHeadMatch(antecedents.get(i), dict)){
	//				ret[i] = true;
	//				found = true;
	//			}
	//		}
	//		return ret;
	//	}  

	public SimpleMatrix getAnaphoricityFeatures(Mention m, Document document,
			Map<Integer, List<Mention>> mentionsByHeadIndex) {
		return NeuralUtils.concatenate(
				getMentionFeatures(m, document, mentionsByHeadIndex),
				encodeGenre(document)
				);
	}

	private SimpleMatrix getMentionFeatures(Mention m, Document document,
			Map<Integer, List<Mention>> mentionsByHeadIndex) {
		return NeuralUtils.concatenate(
				NeuralUtils.oneHot(m.mentionType.ordinal(), 4),
				encodeDistance(m.endIndex - m.startIndex - 1),
				new SimpleMatrix(new double[][] {
						{m.mentionNum / (double) document.predictedMentionsByID.size()},
						{mentionsByHeadIndex.get(m.headIndex).stream()
							.anyMatch(m2 -> m != m2 && m.insideIn(m2)) ? 1 : 0}})
				);
	}

	private static SimpleMatrix encodeDistance(int d) {
		SimpleMatrix m = new SimpleMatrix(11, 1);
		if (d < 5) {
			m.set(d, 1);
		} else if (d < 8) {
			m.set(5, 1);
		} else if (d < 16) {
			m.set(6, 1);
		} else if (d < 32) {
			m.set(7, 1);
		} else if (d < 64) {
			m.set(8, 1);
		} else {
			m.set(9, 1);
		}
		m.set(10, Math.min(d, 64) / 64.0);
		return m;
	}

	private SimpleMatrix encodeGenre(Document document) {
		return conll ? NeuralUtils.oneHot(
				genres.get(document.docInfo.get("DOC_ID").split("/")[0]), genres.size()) :
					new SimpleMatrix(1, 1);
	}

}
