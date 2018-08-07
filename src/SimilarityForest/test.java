package SimilarityForest;


import java.io.BufferedReader;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.*;
import java.util.Random;
import java.util.concurrent.SynchronousQueue;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.trees.ht.GiniSplitMetric;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.unsupervised.attribute.Copy;
import weka.core.EuclideanDistance;



public class test extends J48 implements Classifier {
	private int r = 1;
	
	public test() {
		// TODO Auto-generated constructor stub
	}
	
	
	public static void main(String[] args) {
		try {
			BufferedReader trainBR = new BufferedReader(new FileReader("C:/Users/Administrator/Desktop/Datasets/ieee/Training_norm.arff"));			
			Instances inputData = new Instances(trainBR);
			trainBR.close();
			
			inputData.setClassIndex(inputData.numAttributes() - 1);
			
			BufferedReader testBR = new BufferedReader(new FileReader("C:/Users/Administrator/Desktop/Datasets/ieee/Test_norm.arff"));			
			Instances testData = new Instances(testBR);
			testBR.close();
			
			testData.setClassIndex(testData.numAttributes() - 1);

			Bagging bag = new Bagging();
			bag.setClassifier(new SimTree());
			bag.buildClassifier(inputData);
			bag.setNumIterations(10);
			Evaluation eval = new Evaluation(testData);
			eval.crossValidateModel(bag, testData, 10, new Random(1));

			System.out.println(eval.toSummaryString());
			System.out.println(eval.toMatrixString());
		} catch (Exception e) {
			System.out.println(e);
			 e.printStackTrace(System.out);
		} 		
		
	}	
	
	
	public static double distance(Instance first, Instance second, int numAttributes) {
			double distance = 0;
			double dotprod = 0;
			double norm1 = 0;
			double norm2 = 0;
			for (int k = 1; k<  numAttributes; k++ ) {
				dotprod += first.value(k) * second.value(k);
				if (first.value(k) == ' ' || second.value(k) == ' ') {
					System.out.println(k);
				}
				norm1 += first.value(k) * first.value(k);
				norm2 += second.value(k) * second.value(k);
			}
			distance = dotprod/Math.sqrt(norm1*norm2);
			return distance;
		}

}
