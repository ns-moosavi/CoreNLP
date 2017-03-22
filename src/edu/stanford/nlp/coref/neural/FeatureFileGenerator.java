package edu.stanford.nlp.coref.neural;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import edu.stanford.nlp.coref.CorefDocumentProcessor;
import edu.stanford.nlp.coref.CorefProperties;
import edu.stanford.nlp.coref.CorefProperties.Dataset;
import edu.stanford.nlp.coref.CorefUtils;
import edu.stanford.nlp.coref.data.Dictionaries;
import edu.stanford.nlp.coref.data.Dictionaries.MentionType;
import edu.stanford.nlp.coref.data.Document;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.dcoref.Rules;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.StringUtils;

/**
 * Outputs the CoNLL CoNLL data for training the neural coreference system
 * (implented in python/theano).
 * See <a href="https://github.com/clarkkev/deep-coref">https://github.com/clarkkev/deep-coref</a>
 * for the training code.
 * @author Kevin Clark
 */
public class FeatureFileGenerator implements CorefDocumentProcessor {
	private final PrintWriter dataWriter;
	private final Dictionaries dictionaries;
	public int[] dev_counts;
	public int[] pos_counts, neg_counts;
	public int drop_rate = -1;
	public int drop_cnt = 0;
	private final boolean m_skip_pronominal_pos_antecedents = false;

	public FeatureFileGenerator(Properties props, Dictionaries dictionaries, String dataPath) {


		this.dictionaries = dictionaries;
		try {
			dataWriter = IOUtils.getPrintWriter(dataPath);
			String featureIdentifiers = "class ## ";
			featureIdentifiers += "ana_type ## ana_f_type ## ana_ner ## ana_len ## ana_rel ## ana_f_POS ## ana_l_POS ## ana_head_POS ## ana_n_POS ## ana_nn_POS ## ana_p_POS ## ana_pp_POS ## ";
			featureIdentifiers += "ant_type ## ant_f_type ## ant_ner ## ant_len ## ant_rel ## ant_f_POS ## ant_l_POS ## ant_head_POS ## ant_n_POS ## ant_nn_POS ## ant_p_POS ## ant_pp_POS ## ";
			featureIdentifiers += "exact_match ## head_match ## token_contained ## head_contained ## compatible_modifiers ## first_compatible_head_match ## ";
			featureIdentifiers += "comp_number ## comp_gender ## comp_animacy ## comp_attributes ## f_comp_att ## f_com_obj ## f_comp_subj ## acronym ## distance ## genre" ;

			dataWriter.println(featureIdentifiers);

		} catch (Exception e) {
			throw new RuntimeException("Error creating data exporter", e);
		}
	}

	@Override
	public void process(int id, Document document) {
		String genre = document.docInfo.get("DOC_ID").split("/")[0];
//		if (genre.equals("tc")  || genre.equals("bc")){
			List<Mention> mentionList = CorefUtils.getSortedMentions(document);
			for (int i = 1; i < mentionList.size(); i++){
				Mention m1 = mentionList.get(i);
				List<Mention> allAntecedents = mentionList.subList(0, i);
					if (m1.mentionType == MentionType.LIST)
						writeFeatures(id, document, m1, allAntecedents, document.goldMentionsByID, dictionaries, dataWriter);
			}
//		}
	}

	List<String> getMentionFeatures(Mention m){
		List<String> atts = new ArrayList<String>();
		Iterator<SemanticGraphEdge> iterator =
				m.enhancedDependency.incomingEdgeIterator(m.headIndexedWord);
		SemanticGraphEdge relation = iterator.hasNext() ? iterator.next() : null;
		String depRelation = relation == null ? "no-parent" : relation.getRelation().toString();

		atts.add(m.mentionType.toString());
		atts.add(CategoricalFeatureExtractor.fine_type(m));
		//		atts.add(m.gender.toString());
		//		atts.add(m.number.toString());
		atts.add(m.nerString);
		atts.add(""+m.originalSpan.size());
		atts.add(depRelation);
		atts.add(m.sentenceWords.get(m.startIndex).get(CoreAnnotations.PartOfSpeechAnnotation.class));
		atts.add(m.sentenceWords.get(m.endIndex-1).get(CoreAnnotations.PartOfSpeechAnnotation.class));
		atts.add(m.sentenceWords.get(m.headIndex).get(CoreAnnotations.PartOfSpeechAnnotation.class));
		atts.add(m.endIndex < m.sentenceWords.size() ? m.sentenceWords.get(m.endIndex).get(CoreAnnotations.PartOfSpeechAnnotation.class) : "END");
		atts.add(m.endIndex+1 < m.sentenceWords.size() ? m.sentenceWords.get(m.endIndex+1).get(CoreAnnotations.PartOfSpeechAnnotation.class) : "END");
		atts.add(m.startIndex-1 > 0 ? m.sentenceWords.get(m.startIndex-1).get(CoreAnnotations.PartOfSpeechAnnotation.class) : "START");
		atts.add(m.startIndex-2 > 0 ? m.sentenceWords.get(m.startIndex-2).get(CoreAnnotations.PartOfSpeechAnnotation.class) : "START");

		return atts;
	}

	private void writeFeatures(int docNum, Document document, Mention m, List<Mention> antecedents, 
			Map<Integer, Mention> goldMentions, Dictionaries dict, PrintWriter pw){

		Mention goldM = goldMentions.get(m.mentionID);
		String attLine = "";

		List<Boolean[]> allPairwise = getPairwiseFeatures(document, m, antecedents, dict);


		for (int i = antecedents.size()-1; i >=0; i--){
			drop_cnt++;
			Boolean[] pairwise = allPairwise.get(allPairwise.size()-1 - i); 
			attLine = "";
			Mention ante = antecedents.get(i);

			Mention goldAnte = goldMentions.get(ante.mentionID);

			int label = 0;
			if (goldM != null && goldAnte != null && goldM.goldCorefClusterID == goldAnte.goldCorefClusterID){
				label = docNum * 1000 + goldM.goldCorefClusterID;
				attLine += ""+label+ " ## ";
			}
			else
				attLine += "0 ## ";

			if (label > 0 && ante.mentionType == MentionType.PRONOMINAL && m_skip_pronominal_pos_antecedents)
				continue;

			if (label == 0 && drop_rate > 0 && drop_cnt%drop_rate == 0)
				continue;

			for (String s : getMentionFeatures(m))
				attLine += s + " ## ";

			for (String s : getMentionFeatures(ante))
				attLine += s + " ## ";




			for (Boolean v : pairwise)
				attLine += v ? "True ## " : "False ## ";

			attLine += m.sentNum - ante.sentNum;
			attLine += " ## " + document.docInfo.get("DOC_ID").split("/")[0];

			pw.println(attLine);

		}
	}

	List<Boolean[]> getPairwiseFeatures(Document document, Mention m, List<Mention> allAnte, Dictionaries dict){
		List<Boolean[]> allResults = new ArrayList<Boolean[]>();
		boolean[] firstCompatibles = new boolean[4];
		for (int i = allAnte.size()-1; i >=0; i--){

			Mention ante = allAnte.get(i);			
			Boolean[] res = new Boolean[14];
			int cnt = 0;
			boolean exact_match = false, head_match = false, token_contained = false, head_contained = false;

			Iterator<SemanticGraphEdge> iterator =
					ante.enhancedDependency.incomingEdgeIterator(ante.headIndexedWord);
			SemanticGraphEdge relation = iterator.hasNext() ? iterator.next() : null;
			String ant_depRelation = relation == null ? "no-parent" : relation.getRelation().toString();
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
			res[cnt++] = exact_match;
			res[cnt++] = head_match;
			res[cnt++] = token_contained;
			res[cnt++] = head_contained;
			res[cnt++] = m.compatibleModifiers(ante);

			if (head_match && !firstCompatibles[0] && m.compatibleModifiers(ante)){
				res[cnt] = true;
				firstCompatibles[0] = true;
			}
			else
				res[cnt] = false;
			cnt++;

			if (m.numbersAgree(ante))
				res[cnt] = true;
			else
				res[cnt] = false;
			cnt++;

			if (m.gendersAgree(ante))
				res[cnt] = true;
			else
				res[cnt] = false;
			cnt++;

			if (m.animaciesAgree(ante))
				res[cnt] = true;
			else
				res[cnt] = false;
			cnt++;

			boolean compatible = false;
			if (m.numbersAgree(ante) && m.gendersAgree(ante) && m.animaciesAgree(ante)){
				compatible = true;
			}

			if (compatible)
				res[cnt] = true;
			else
				res[cnt] = false;
			cnt++;


			if (compatible && !firstCompatibles[1]){
				res[cnt] = true;
				firstCompatibles[1] = true;
			}
			else
				res[cnt] = false;
			cnt++;

			if (!firstCompatibles[2] && (ant_depRelation.startsWith("dobj") || ant_depRelation.startsWith("iobj"))){
				res[cnt] = true;
				firstCompatibles[2] =true;
			}
			else
				res[cnt] = false;
			cnt++;

			if (!firstCompatibles[3] && ant_depRelation.startsWith("nsubj")){
				res[cnt] = true;
				firstCompatibles[3] = true;
			}
			else
				res[cnt] = false;
			cnt++;

			if (Rules.isAcronym(m.originalSpan, ante.originalSpan))
				res[cnt] = true;
			else
				res[cnt] = false;
			cnt++;


			allResults.add(res);
		}

		return allResults;
	}


	@Override
	public void finish() throws Exception {
		dataWriter.close();
	}


	public static void exportData(String outputPath, Dataset dataset, Properties props,
			Dictionaries dictionaries) throws Exception {
		CorefProperties.setInput(props, dataset);
		String dataPath = outputPath + "/attribute_files/";
		IOUtils.ensureDir(new File(outputPath));
		IOUtils.ensureDir(new File(dataPath));
		FeatureFileGenerator dataExporter = new FeatureFileGenerator(props, dictionaries,
				dataPath + dataset.toString().toLowerCase()+"_list");

		dataExporter.run(props, dictionaries);

		return;
	}

	public static void main(String[] args) throws Exception {
		Properties props = StringUtils.argsToProperties(new String[] {"-props", args[0]});
		Dictionaries dictionaries = new Dictionaries(props);
		String outputPath = args[1];
		//		int[] dev_feat_count = exportData(outputPath, Dataset.DEV, props, dictionaries, null);

		exportData(outputPath, Dataset.TRAIN, props, dictionaries);
		exportData(outputPath, Dataset.DEV, props, dictionaries);
		//		System.out.println(dev_feat_count.length);
		//				exportData(outputPath, Dataset.TEST, props, dictionaries, null);
		//		System.out.println(dev_feat_count.length);
		//		System.out.println();
	}

}
