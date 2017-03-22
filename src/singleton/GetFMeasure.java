package singleton;

public class GetFMeasure {
	public static void main(String[] args){
		double recall =80.45, precision = 83.76;
		System.out.println((recall*2*precision)/(recall+precision));
		
	}

}
