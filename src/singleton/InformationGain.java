package singleton;

import java.util.HashMap;
import java.util.Map;

public class InformationGain {

	private class Pair{
		int a,b;
		public Pair(int a , int b) {
			this.a = a;
			this.b = b;
		}
	}
	
	private static double LOG_FACTOR = 1.0 / Math.log(2);
	private Map<Pair, Double> m_infGainCache = new HashMap<Pair, Double>();
	private Map<Pair, Double> m_likelihoodCache = new HashMap<Pair, Double>();


	private double minimalGain = Math.pow(10, -5);
	private boolean useDirectedLikelihood = false;

	public InformationGain() {}

	public InformationGain(double minimalGain) {
		this.minimalGain = minimalGain;
	}

	public void setMinimalGain(double minimalGain) {
		this.minimalGain = minimalGain;    	
	}
	

	public double getBenefit(double k11, double k12, double k21, double k22) {
		Pair p = new Pair((int)k11, (int)k12);
		if (m_infGainCache.containsKey(p))
			return m_infGainCache.get(p);
		double[][] weightCounts = {{k11, k12},{k21,k22}};
		int numberOfValues = weightCounts.length;
		int numberOfLabels = weightCounts[0].length;

		// calculate entropies
		double[] entropies = new double[numberOfValues];
		double[] totalWeights = new double[numberOfValues]; 
		for (int v = 0; v < numberOfValues; v++) {
			for (int l = 0; l < numberOfLabels; l++) {
				totalWeights[v] += weightCounts[v][l];
			}

			for (int l = 0; l < numberOfLabels; l++) {
				if (weightCounts[v][l] > 0) {
					double proportion = weightCounts[v][l] / totalWeights[v];
					entropies[v] -= (Math.log(proportion) * LOG_FACTOR) * proportion;
				}
			}
		}

		// calculate information amount WITH this attribute
		double totalWeight = 0.0d;
		for (double w : totalWeights) {
			totalWeight += w;
		}

		double information = 0.0d;
		for (int v = 0; v < numberOfValues; v++) {
			information += totalWeights[v] / totalWeight * entropies[v];
		}


		// calculate information amount WITHOUT this attribute
		double[] classWeights = new double[numberOfLabels];
		for (int l = 0; l < numberOfLabels; l++) {
			for (int v = 0; v < numberOfValues; v++) {
				classWeights[l] += weightCounts[v][l];
			}
		}

		double totalClassWeight = 0.0d;
		for (double w : classWeights) {
			totalClassWeight += w;
		}

		double classEntropy = 0.0d;
		for (int l = 0; l < numberOfLabels; l++) {
			if (classWeights[l] > 0) {
				double proportion = classWeights[l] / totalClassWeight;
				classEntropy -= (Math.log(proportion) * LOG_FACTOR) * proportion;
			}
		}

		// calculate and return information gain
		double informationGain = classEntropy - information;
		if (informationGain < minimalGain * classEntropy) {
			informationGain = 0;
		}

		m_infGainCache.put(p, informationGain);
		return informationGain;
	}


	public double getBenefit(double[][] weightCounts) {
		int numberOfValues = weightCounts.length;
		int numberOfLabels = weightCounts[0].length;

		// calculate entropies
		double[] entropies = new double[numberOfValues];
		double[] totalWeights = new double[numberOfValues]; 
		for (int v = 0; v < numberOfValues; v++) {
			for (int l = 0; l < numberOfLabels; l++) {
				totalWeights[v] += weightCounts[v][l];
			}

			for (int l = 0; l < numberOfLabels; l++) {
				if (weightCounts[v][l] > 0) {
					double proportion = weightCounts[v][l] / totalWeights[v];
					entropies[v] -= (Math.log(proportion) * LOG_FACTOR) * proportion;
				}
			}
		}

		// calculate information amount WITH this attribute
		double totalWeight = 0.0d;
		for (double w : totalWeights) {
			totalWeight += w;
		}

		double information = 0.0d;
		for (int v = 0; v < numberOfValues; v++) {
			information += totalWeights[v] / totalWeight * entropies[v];
		}


		// calculate information amount WITHOUT this attribute
		double[] classWeights = new double[numberOfLabels];
		for (int l = 0; l < numberOfLabels; l++) {
			for (int v = 0; v < numberOfValues; v++) {
				classWeights[l] += weightCounts[v][l];
			}
		}

		double totalClassWeight = 0.0d;
		for (double w : classWeights) {
			totalClassWeight += w;
		}

		double classEntropy = 0.0d;
		for (int l = 0; l < numberOfLabels; l++) {
			if (classWeights[l] > 0) {
				double proportion = classWeights[l] / totalClassWeight;
				classEntropy -= (Math.log(proportion) * LOG_FACTOR) * proportion;
			}
		}

		// calculate and return information gain
		double informationGain = classEntropy - information;
		if (informationGain < minimalGain * classEntropy) {
			informationGain = 0;
		}
		return informationGain;
	}

	protected double getEntropy(double[] labelWeights, double totalWeight) {
		double entropy = 0;
		for (int i = 0; i < labelWeights.length; i++) {
			if (labelWeights[i] > 0) {
				double proportion = labelWeights[i] / totalWeight;
				entropy -= (Math.log(proportion) * LOG_FACTOR) * proportion;
			}
		}
		return entropy;
	}


	private double log(double x){
		double ans = 0;
		if (x > 0)
			ans = Math.log(x);
		return ans;
	}

	public double InformationGainFormula(double theta, double p, double q){
		if (theta==0 || q < 0)
			return 0;

		double conditionalProb_term1 = -theta*q*(log(q))-theta*(1-q)*(log(1-q));
		double conditionalProb_term2 = (theta*q-p)*(log(p-theta*q) - log(1-theta));
		double conditionalProb_term3 = (theta*(1-q)-(1-p))*(log(((1-p)-(theta*(1-q)))/(1-theta)));

		double conditionalProb = conditionalProb_term1 + conditionalProb_term2 + conditionalProb_term3;

		double nonCondProb = -p*(log(p))-(1-p)*(log(1-p));
		double informationGain = nonCondProb - conditionalProb;
		if (informationGain < minimalGain * nonCondProb)
			informationGain = 0;
		return informationGain;
	}

	public double informationGainUpperBound(double theta, double p){
		double upperBound = 0;
		if (theta <= p){
			upperBound = InformationGainFormula(theta,p,1);
		}
		else {
			upperBound = InformationGainFormula(theta,p,p/theta);
			//			System.out.println("Upper: " + InformationGainFormula(theta,p,p/theta) + " " + InformationGainFormula(theta, p, 1-((1-p)/theta)));
		}

		return upperBound;
	}

	public static void main(String[] args){
		InformationGain ig = new InformationGain();
		double[][] a = {{30,120},{70,300}};
		System.out.println(ig.getBenefit(a));
	}

	double entropy(double... elements) {
		double sum = 0;
		for (double element : elements) {
			sum += element;
		}
		double result = 0.0;
		for (double x : elements) {
			if (x < 0) {
				throw new IllegalArgumentException("Should not have negative count for entropy computation: (" + x + ')');
			}

			int zeroFlag = (x == 0 ? 1 : 0);
			result += x * log((x + zeroFlag) / sum);
		}
		return -result;
	}

	public double logLikelihoodRatio(double k11, double k12, double k21, double k22) {
		Pair p = new Pair((int)k11, (int)k12);
		if (m_likelihoodCache.containsKey(p))
			return m_likelihoodCache.get(p);

		double rowEntropy = entropy(k11, k12) + entropy(k21, k22);
		double columnEntropy = entropy(k11, k21) + entropy(k12, k22);
		double matrixEntropy = entropy(k11, k12, k21, k22);
		if (rowEntropy + columnEntropy > matrixEntropy) {
			// round off error
			return 0.0;
		}
		double llr = 2.0 * (matrixEntropy - rowEntropy - columnEntropy);
		if (useDirectedLikelihood){
			double e11 = (k11+k12)*(k11+k21)/(k11+k12+k21+k22);
			if (k11 < e11)
				llr *= -1;
		}

		llr = Math.floor(llr * 100) / 100;
		m_likelihoodCache.put(p, llr);
		return llr;
	}

}
