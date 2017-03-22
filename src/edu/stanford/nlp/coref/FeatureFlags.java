package edu.stanford.nlp.coref;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.Properties;

import edu.stanford.nlp.process.WordShapeClassifier;

public class FeatureFlags implements Serializable  {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public static final String DEFAULT_BACKGROUND_SYMBOL = "O";

	private String stringRep = "";


	public boolean useNGrams = false;
	public boolean conjoinShapeNGrams = false;
	public boolean lowercaseNGrams = false;
	public boolean dehyphenateNGrams = false;
	public boolean usePrev = false;
	public boolean useNext = false;
	public boolean useTags = false;
	public boolean useWordPairs = false;
	public boolean useGazettes = false;
	public boolean useNeighborNGrams = false;
	public boolean useText = false;
	public boolean useLemma = false;
	public boolean useNewInnerWordRepresentation = false;
	public boolean useLowerCasedStrings = false;
	public boolean useWholeNPText = false;
	public boolean useWholeNPLemma = false;
	public boolean useWholeNPTags = false;
	public boolean useHead = false;
	public boolean useFeaturePruning = false;
	public boolean useVerbs = false;
	public boolean useNERTags = false;
	public boolean useLength = false;
	public boolean usePosition = false;
	public boolean useLinguistic = false;
	public boolean useInfoGain = false;
	public boolean greekifyNGrams = false;
	public boolean noMidNGrams = false;
	public boolean intern = false;
	public boolean prunTC = false;
	public boolean useNPType = false;
	public boolean useHeadMatch = false;
	public boolean useTextMatch = false;

	
	public int windowSize = 0;
	public int prunThreshold = 0;
	public int maxMentionLength =-1;
	public int maxNGramLeng = -1;

	public int wordShape = WordShapeClassifier.NOWORDSHAPE;
	
	public String trainDir = "";
	public String devDir="";
	public String testDir = "";
	public String featureFile="";
	public String outFile = "";

	public transient Properties props = null;


	public FeatureFlags(Properties props) {
		setProperties(props, true);
	}

	public void setProperties(Properties props, boolean printProps) {
		this.props = props;
		StringBuilder sb = new StringBuilder(stringRep);
		for (Enumeration e = props.propertyNames(); e.hasMoreElements();) {
			String key = (String) e.nextElement();
			String val = props.getProperty(key);
			if (!(key.length() == 0 && val.length() == 0)) {
				if (printProps) {
					System.err.println(key + '=' + val);
				}
				sb.append(key).append('=').append(val).append('\n');
			}
			
			if (key.equalsIgnoreCase("useNGrams")) {
				useNGrams = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useNeighborNGrams")) {
				useNeighborNGrams = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("conjoinShapeNGrams")) {
				conjoinShapeNGrams = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("lowercaseNGrams")) {
				lowercaseNGrams = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useText")) {
				useText = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useTags")) {
				useTags = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useLemma")) {
				useLemma = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useLowerCasedStrings")) {
				useLowerCasedStrings = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useWholeNPText")) {
				useWholeNPText = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useWholeNPLemma")) {
				useWholeNPLemma = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useWholeNPTags")) {
				useWholeNPTags = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useFeaturePruning")) {
				useFeaturePruning = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useHead")) {
				useHead = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useVerbs")) {
				useVerbs = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useNERTags")) {
				useNERTags = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useLinguistic")) {
				useLinguistic = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("usePosition")) {
				usePosition = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useInfoGain")) {
				useInfoGain = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useLength")) {
				useLength = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("greekifyNGrams")) {
				greekifyNGrams = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("intern")) {
				intern = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("noMidNGrams")) {
				noMidNGrams = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("prunTC")) {
				prunTC = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useNPType")) {
				useNPType = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useHeadMatch")) {
				useHeadMatch = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useTextMatch")) {
				useTextMatch = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("useNewInnerWordRepresentation")) {
				useNewInnerWordRepresentation = Boolean.parseBoolean(val);
			} else if (key.equalsIgnoreCase("windowSize")) {
				windowSize = Integer.parseInt(val);
			} else if (key.equalsIgnoreCase("prunThreshold")) {
				prunThreshold = Integer.parseInt(val);
			} else if (key.equalsIgnoreCase("maxNGramLeng")) {
				maxNGramLeng = Integer.parseInt(val);
			}  else if (key.equalsIgnoreCase("trainDir")) {
				trainDir = val;
			} else if (key.equalsIgnoreCase("testDir")) {
				testDir = val;
			} else if (key.equalsIgnoreCase("devDir")) {
				devDir = val;
			} else if (key.equalsIgnoreCase("featureFile")) {
				featureFile = val;
			} else if (key.equalsIgnoreCase("outFile")) {
				outFile = val;
			}
		}
	}
}
