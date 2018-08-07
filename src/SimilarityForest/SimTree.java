/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * SimpleCart.java
 * Copyright (C) 2007 Haijian Shi
 *
 */

package SimilarityForest;


import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.RandomizableClassifier;
import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;



import java.util.ArrayList;

import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.Vector;

/**
 <!-- globalinfo-start -->
 * Class implementing minimal cost-complexity pruning.<br/>
 * Note when dealing with missing values, use "fractional instances" method instead of surrogate split method.<br/>
 * <br/>
 * For more information, see:<br/>
 * <br/>
 * Leo Breiman, Jerome H. Friedman, Richard A. Olshen, Charles J. Stone (1984). Classification and Regression Trees. Wadsworth International Group, Belmont, California.
 * <p/>
 <!-- globalinfo-end -->	
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;book{Breiman1984,
 *    address = {Belmont, California},
 *    author = {Leo Breiman and Jerome H. Friedman and Richard A. Olshen and Charles J. Stone},
 *    publisher = {Wadsworth International Group},
 *    title = {Classification and Regression Trees},
 *    year = {1984}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 * <pre> -M &lt;min no&gt;
 *  The minimal number of instances at the terminal nodes.
 *  (default 2)</pre>
 * 
 * <pre> -N &lt;num folds&gt;
 *  The number of folds used in the minimal cost-complexity pruning.
 *  (default 5)</pre>
 * 
 * <pre> -U
 *  Don't use the minimal cost-complexity pruning.
 *  (default yes).</pre>
 * 
 * <pre> -H
 *  Don't use the heuristic method for binary split.
 *  (default true).</pre>
 * 
 * <pre> -A
 *  Use 1 SE rule to make pruning decision.
 *  (default no).</pre>
 * 
 * <pre> -C
 *  Percentage of training data size (0-1].
 *  (default 1).</pre>
 * 
 <!-- options-end -->
 *
 * @author Haijian Shi (hs69@cs.waikato.ac.nz)
 * @version $Revision: 8109 $
 */
public class SimTree
  extends RandomizableClassifier
  implements AdditionalMeasureProducer, TechnicalInformationHandler {

  /** For serialization.	 */
  private static final long serialVersionUID = 4154189200352566053L;

  /** Training data.  */
  protected Instances m_train;

  /** Successor nodes. */
  protected SimTree[] m_Successors;

  /** Attribute used to split data. */
  protected Attribute m_Attribute;

  /** Split point for a numeric attribute. */
  protected double m_SplitValue;

  /** Split subset used to split data for nominal attributes. */
  protected String m_SplitString;

  /** Class value if the node is leaf. */
  protected double m_ClassValue;

  /** Class attribute of data. */
  protected Attribute m_ClassAttribute;

  /** Minimum number of instances in at the terminal nodes. */
  protected double m_minNumObj = 1;

  /** Number of folds for minimal cost-complexity pruning. */
  protected int m_numFoldsPruning = 5;

  /** Alpha-value (for pruning) at the node. */
  protected double m_Alpha;

  /** Number of training examples misclassified by the model (subtree rooted). */
  protected double m_numIncorrectModel;

  /** Number of training examples misclassified by the model (subtree not rooted). */
  protected double m_numIncorrectTree;

  /** Indicate if the node is a leaf node. */
  protected boolean m_isLeaf;

  /** If use minimal cost-compexity pruning. */
  protected boolean m_Prune = false;

  /** Total number of instances used to build the classifier. */
  protected int m_totalTrainInstances;

  /** Proportion for each branch. */
  protected double[] m_Props;

  /** Class probabilities. */
  protected double[] m_ClassProbs = null;

  /** Distributions of leaf node (or temporary leaf node in minimal cost-complexity pruning) */
  protected double[] m_Distribution;

  /** If use huristic search for nominal attributes in multi-class problems (default true). */
  protected boolean m_Heuristic = false;

  /** If use the 1SE rule to make final decision tree. */
  protected boolean m_UseOneSE = false;

  /** Training data size. */
  protected double m_SizePer = 1;
  
  /** Training data size. */
  protected double m_splitValue = 1;
  
  /** Training data size. */
  protected Instance obi, obj;
  
  protected Instances sample;
  
  protected List<Double> nonComputablePrediction;

  /**
   * Return a description suitable for displaying in the explorer/experimenter.
   * 
   * @return 		a description suitable for displaying in the 
   * 			explorer/experimenter
   */
  public String globalInfo() {
    return  
        "Similarity Tree Computation.\n"
      + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return 		the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    
    result = new TechnicalInformation(Type.CONFERENCE);
    result.setValue(Field.AUTHOR, "Sachit Sathe and Charu C. Aggarwal");
    result.setValue(Field.YEAR, "2017");
    result.setValue(Field.TITLE, "Similarity Forests");
    result.setValue(Field.PUBLISHER, "ACM");
    result.setValue(Field.ADDRESS, "New York");
    
    return result;
  }

  /**
   * Returns default capabilities of the classifier.
   * 
   * @return 		the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);

    return result;
  }

  /**
   * Build the classifier.
   * 
   * @param data 	the training instances
   * @throws Exception 	if something goes wrong
   */
  public void buildClassifier(Instances data) throws Exception {

    getCapabilities().testWithFail(data);
    data = new Instances(data);        
    data.deleteWithMissingClass();
    makeTree(data);
    return;
   
  }

  /**
   * Make binary decision tree recursively.
   * 
   * @param data 		the training instances
   * @param totalInstances 	total number of instances
   * @param sortedIndices 	sorted indices of the instances
   * @param weights 		weights of the instances
   * @param classProbs 		class probabilities
   * @param totalWeight 	total weight of instances
   * @param minNumObj 		minimal number of instances at leaf nodes
   * @param useHeuristic 	if use heuristic search for nominal attributes in multi-class problem
   * @throws Exception 		if something goes wrong
   */
  protected void makeTree(Instances data) throws Exception{

	  m_totalTrainInstances = data.numInstances();
	  m_isLeaf = true;
	  m_ClassAttribute = data.classAttribute();
	  m_Attribute = new Attribute("Order", false);
	  nonComputablePrediction = new ArrayList<Double>();
	  
	  sample = new Instances(data, 0);
	  // Check if node does not contain enough instances, or if it can not be split,
	  // or if it is pure. If does, make leaf.
	  
	  if (m_totalTrainInstances < 2 * m_minNumObj || (data.numDistinctValues(data.classIndex()) < 2)) {
		  makeLeaf(data);
	  } else {          
		  Instances[] splitInstances = new Instances[2];	
		  Instances[] cleanSplit = new Instances[2];
		  
		  splitData(splitInstances, data);
		  if (splitInstances[0] == null || splitInstances[1] == null) {
			  	m_Attribute = null;
			    m_isLeaf = true;
			    m_ClassValue= getHighestPredCount(nonComputablePrediction)[0] == 1.00 ? 0.00: 1.00;
			    m_ClassAttribute = data.classAttribute();
			  return;
		  }
		  m_Props = new double[]{splitInstances[0].numInstances(), splitInstances[1].numInstances()};
		  
		  m_SplitValue = (splitInstances[0].get(splitInstances[0].numInstances() - 1).value(0)
				  + splitInstances[0].get(splitInstances[0].numInstances() - 1).value(1))/2;

		  Remove removef = new Remove();
		  removef.setAttributeIndices("first");
		  removef.setInvertSelection(false);
		  removef.setInputFormat(splitInstances[0]);
		  
		  cleanSplit[0] = Filter.useFilter(splitInstances[0], removef);
		  Remove removes = new Remove();
		  removes.setAttributeIndices("first");
		  removes.setInvertSelection(false);
		  removes.setInputFormat(splitInstances[1]);
		  
		  cleanSplit[1] = Filter.useFilter(splitInstances[1], removes);

		  m_isLeaf = false;
		  m_Successors = new SimTree[2];
		  for (int i = 0; i < 2; i++) {
				m_Successors[i] = new SimTree();
				m_Successors[i].makeTree(cleanSplit[i]);
		  }
    }
		
  }
 

  /**
   * Split data into two subsets and store sorted indices and weights for two
   * successor nodes.
   * 
   * @param subsetIndices 	sorted indecis of instances for each attribute 
   * 				for two successor node
   * @param subsetWeights 	weights of instances for each attribute for 
   * 				two successor node
   * @param att 		attribute the split based on
   * @param splitPoint 		split point the split based on if att is numeric
   * @param splitStr 		split subset the split based on if att is nominal
   * @param sortedIndices 	sorted indices of the instances to be split
   * @param weights 		weights of the instances to bes split
   * @param data 		training data
   * @throws Exception 		if something goes wrong  
   */
  protected void splitData(Instances[] splitInstances, Instances data) throws Exception {
	int i, j;
	double o;
	double gini1, gini2, wgini;
	double bestGini = 1;
	int bestSplit = 0;
	Instance ob1 = null, ob2 = null;
	// for r number of random pairs
	// Create 
	Instances[] bestInstances = new Instances[2];
	for (int r = 0; r < 1; r++) {
		try {
			ArrayList<Attribute> aList = new ArrayList<Attribute>();
			aList.add(new Attribute("Order"));
			Instances ord = new Instances("Order", aList, data.numInstances());
			randomPair(data);
			for (i = 0; i < data.numInstances(); i++) {
				o = cosSim(data.instance(i), obj, data.numAttributes() - 1) - cosSim(data.instance(i), obi, data.numAttributes() - 1);
				if (o < -5) {
					nonComputablePrediction.add(data.instance(i).value(data.numAttributes() - 1));
				}
				double[] ll = new double[]{o};
				ord.add(new DenseInstance(1.0, ll));	
			}
			
			Instances tempData = Instances.mergeInstances(ord, data);
			Instances orderedData = new Instances(tempData, nonComputablePrediction.size(), tempData.numInstances() -  nonComputablePrediction.size() - 1);
			
			
			
			orderedData.sort(0);
			orderedData.setClassIndex(orderedData.numAttributes() - 1);
			String[] classes = new String[orderedData.numInstances()];
			int[] classcount = new int[orderedData.numInstances()];
			HashMap<String, Integer> classMap = new HashMap<String, Integer>();
			
			for (i = 0; i < orderedData.classAttribute().numValues() ; i++) {
				classMap.put(orderedData.classAttribute().value(i), 0);	
			}
			
			
			// Setup to calculate weighted gini index
			for (i = 0; i < orderedData.numInstances(); i++) {
				classes[i] =  orderedData.instance(i).stringValue(orderedData.numAttributes() - 1);
				classMap.put(classes[i], 0);
				classcount[i] = classMap.get(classes[i]);				
			}
			// 
			Set<String> cl = classMap.keySet();
			for (i = 0; i < orderedData.numInstances() - 1; i++) {
				gini1 = 0;
				gini2 = 0;
				int index = i;
				while (index >= 0) {
					classMap.put(classes[index], classMap.get(classes[index]) + 1);
					index--;
				} 
				for (String s: cl) {
					double p = classMap.get(s)/ (double)(i + 1);
					gini1 += (p*p);
					classMap.put(s, 0);
					
				}
					
				//Evaluate lower half of split
				index = i+1;
				while (index < orderedData.numInstances()) {
					classMap.put(classes[index], classMap.get(classes[index]) + 1);
					index++;
				} 
				for (String s: cl) {
					double p = classMap.get(s)/ (double)(orderedData.numInstances() - i -1);
					gini2 += (p*p);
					classMap.put(s, 0);
				}
	
				gini1 = 1- gini1;
				gini2 = 1- gini2;
				if (gini1 < 0 || gini2 < 0) {
					System.out.println(gini1 + " " + gini2);
				}
				wgini = ((i+1) * gini1 + (orderedData.numInstances() - i - 1) * gini2)/ (orderedData.numInstances());
	
				if (wgini < bestGini) {
					if (bestInstances[0] != null && bestInstances[1] != null) {
						bestInstances[0].clear();
						bestInstances[1].clear();
					}
					ob1 = obi;
					ob2 = obj;
					bestSplit = i;
					bestGini = wgini;
					bestInstances[0] = new Instances(orderedData, 0, bestSplit+1);
					bestInstances[1] = new Instances(orderedData, bestSplit+1, orderedData.numInstances()- 1 - bestSplit);
				}
				
						
			}
		} catch (Exception e) {
			System.out.println(nonComputablePrediction.size());
			System.out.println(data.numInstances() - 1);
			System.out.println("-----------");
			System.out.println(e.getMessage());
		}
		
	}

	splitInstances[0] = bestInstances[0];
	splitInstances[1] = bestInstances[1];
	
	obi = ob1;
	obj = ob2;
		
	
  }
  
  public void randomPair (Instances data) {
		Random rand = new Random();
		int n1 = 0;
		int n2 = 0;
		Instance ob1, ob2;
		while(true) {
			n1 = rand.nextInt(data.numInstances());
			n2 = rand.nextInt(data.numInstances());
			if (n1 != n2 && data.instance(n1).classValue() != data.instance(n2).classValue()) {
				ob1 = data.instance(n1);
				ob2 = data.instance(n2);
				break;
			}
		}
		obi = ob1;
		obj = ob2;
	}


  /**
   * Updates the numIncorrectModel field for all nodes when subtree (to be 
   * pruned) is rooted. This is needed for calculating the alpha-values.
   * 
   * @throws Exception 	if something goes wrong
   */
  public void modelErrors() throws Exception{
    Evaluation eval = new Evaluation(m_train);

    if (!m_isLeaf) {
      m_isLeaf = true; //temporarily make leaf

      // calculate distribution for evaluation
      eval.evaluateModel(this, m_train);
      m_numIncorrectModel = eval.incorrect();

      m_isLeaf = false;

      for (int i = 0; i < m_Successors.length; i++)
	m_Successors[i].modelErrors();

    } else {
      eval.evaluateModel(this, m_train);
      m_numIncorrectModel = eval.incorrect();
    }       
  }

  /**
   * Updates the numIncorrectTree field for all nodes. This is needed for
   * calculating the alpha-values.
   * 
   * @throws Exception 	if something goes wrong
   */
  public void treeErrors() throws Exception {
    if (m_isLeaf) {
      m_numIncorrectTree = m_numIncorrectModel;
    } else {
      m_numIncorrectTree = 0;
      for (int i = 0; i < m_Successors.length; i++) {
	m_Successors[i].treeErrors();
	m_numIncorrectTree += m_Successors[i].m_numIncorrectTree;
      }
    }
  }

  /**
   * Updates the alpha field for all nodes.
   * 
   * @throws Exception 	if something goes wrong
   */
  public void calculateAlphas() throws Exception {

    if (!m_isLeaf) {
      double errorDiff = m_numIncorrectModel - m_numIncorrectTree;
      if (errorDiff <=0) {
	//split increases training error (should not normally happen).
	//prune it instantly.
	makeLeaf(m_train);
	m_Alpha = Double.MAX_VALUE;
      } else {
	//compute alpha
	errorDiff /= m_totalTrainInstances;
	m_Alpha = errorDiff / (double)(numLeaves() - 1);
	long alphaLong = Math.round(m_Alpha*Math.pow(10,10));
	m_Alpha = (double)alphaLong/Math.pow(10,10);
	for (int i = 0; i < m_Successors.length; i++) {
	  m_Successors[i].calculateAlphas();
	}
      }
    } else {
      //alpha = infinite for leaves (do not want to prune)
      m_Alpha = Double.MAX_VALUE;
    }
  }

  /**
   * Find the node with minimal alpha value. If two nodes have the same alpha, 
   * choose the one with more leave nodes.
   * 
   * @param nodeList 	list of inner nodes
   * @return 		the node to be pruned
   */
  protected SimTree nodeToPrune(Vector nodeList) {
    if (nodeList.size()==0) return null;
    if (nodeList.size()==1) return (SimTree)nodeList.elementAt(0);
    SimTree returnNode = (SimTree)nodeList.elementAt(0);
    double baseAlpha = returnNode.m_Alpha;
    for (int i=1; i<nodeList.size(); i++) {
      SimTree node = (SimTree)nodeList.elementAt(i);
      if (node.m_Alpha < baseAlpha) {
	baseAlpha = node.m_Alpha;
	returnNode = node;
      } else if (node.m_Alpha == baseAlpha) { // break tie
	if (node.numLeaves()>returnNode.numLeaves()) {
	  returnNode = node;
	}
      }
    }
    return returnNode;
  }

  /**
   * Compute sorted indices, weights and class probabilities for a given 
   * dataset. Return total weights of the data at the node.
   * 
   * @param data 		training data
   * @param sortedIndices 	sorted indices of instances at the node
   * @param weights 		weights of instances at the node
   * @param classProbs 		class probabilities at the node
   * @return total 		weights of instances at the node
   * @throws Exception 		if something goes wrong
   */
  protected double computeSortedInfo(Instances data, int[][] sortedIndices, double[][] weights,
      double[] classProbs) throws Exception {

    // Create array of sorted indices and weights
    double[] vals = new double[data.numInstances()];
    for (int j = 0; j < data.numAttributes(); j++) {
      if (j==data.classIndex()) continue;
      weights[j] = new double[data.numInstances()];

      if (data.attribute(j).isNominal()) {

	// Handling nominal attributes. Putting indices of
	// instances with missing values at the end.
	sortedIndices[j] = new int[data.numInstances()];
	int count = 0;
	for (int i = 0; i < data.numInstances(); i++) {
	  Instance inst = data.instance(i);
	  if (!inst.isMissing(j)) {
	    sortedIndices[j][count] = i;
	    weights[j][count] = inst.weight();
	    count++;
	  }
	}
	for (int i = 0; i < data.numInstances(); i++) {
	  Instance inst = data.instance(i);
	  if (inst.isMissing(j)) {
	    sortedIndices[j][count] = i;
	    weights[j][count] = inst.weight();
	    count++;
	  }
	}
      } else {

	// Sorted indices are computed for numeric attributes
	// missing values instances are put to end 
	for (int i = 0; i < data.numInstances(); i++) {
	  Instance inst = data.instance(i);
	  vals[i] = inst.value(j);
	}
	sortedIndices[j] = Utils.sort(vals);
	for (int i = 0; i < data.numInstances(); i++) {
	  weights[j][i] = data.instance(sortedIndices[j][i]).weight();
	}
      }
    }

    // Compute initial class counts
    double totalWeight = 0;
    for (int i = 0; i < data.numInstances(); i++) {
      Instance inst = data.instance(i);
      classProbs[(int)inst.classValue()] += inst.weight();
      totalWeight += inst.weight();
    }

    return totalWeight;
  }

  /**
   * Compute and return gini gain for given distributions of a node and its 
   * successor nodes.
   * 
   * @param parentDist 	class distributions of parent node
   * @param childDist 	class distributions of successor nodes
   * @return 		Gini gain computed
   */
  protected double computeGiniGain(double[] parentDist, double[][] childDist) {
    double totalWeight = Utils.sum(parentDist);
    if (totalWeight==0) return 0;

    double leftWeight = Utils.sum(childDist[0]);
    double rightWeight = Utils.sum(childDist[1]);

    double parentGini = computeGini(parentDist, totalWeight);
    double leftGini = computeGini(childDist[0],leftWeight);
    double rightGini = computeGini(childDist[1], rightWeight);

    return parentGini - leftWeight/totalWeight*leftGini -
    rightWeight/totalWeight*rightGini;
  }

  /**
   * Compute and return gini index for a given distribution of a node.
   * 
   * @param dist 	class distributions
   * @param total 	class distributions
   * @return 		Gini index of the class distributions
   */
  protected double computeGini(double[] dist, double total) {
    if (total==0) return 0;
    double val = 0;
    for (int i=0; i<dist.length; i++) {
      val += (dist[i]/total)*(dist[i]/total);
    }
    return 1- val;
  }

  /**
   * Computes class probabilities for instance using the decision tree.
   * 
   * @param instance 	the instance for which class probabilities is to be computed
   * @return 		the class probabilities for the given instance
   * @throws Exception 	if something goes wrong
   */
  public double[] distributionForInstance(Instance instance)
		  throws Exception {
	  if (!m_isLeaf) {
		  double o = cosSim(instance, obj, instance.numAttributes() - 1) - cosSim(instance, obi, instance.numAttributes() - 1);
		  if (o < -5 ) {
			  if (nonComputablePrediction.isEmpty()) {
					return instance.value(instance.numAttributes() - 1) == 0.00 
							? new double[]{0.00, 1.00}
							: new double[]{1.00, 0.00};
				} else {
					return getHighestPredCount(nonComputablePrediction);
				}
		  } else {
			  if (o <= m_SplitValue){
				  return m_Successors[0].distributionForInstance(instance); 
			  }
			  else {
				  return m_Successors[1].distributionForInstance(instance);
			  }  
		  }
		  		  
    }
    else {
		double d[] = new double[2];
    	if (m_ClassValue == 0.00){
    		
    		d[0] = 1.0;
			d[1] = 0.0;
			return d;
    	}
		else {
			d[0] = 0.0;
			d[1] = 1.0;
			return d;
		}
    }
  }
  
  public double[] getHighestPredCount (List<Double> arr){
	  @SuppressWarnings("rawtypes")
	HashMap hm = new HashMap();
	  hm.put(0.00, 0);
	  hm.put(1.00, 0);
	    for (int i = 0; i < arr.size(); i++) {
	        double key = arr.get(i);
            int value =(int) hm.get(key);
            hm.put(key, value + 1);
	    }
	    return ((int)hm.get(0.00) >=  (int)hm.get(1.00))
	    		? new double[]{1.00, 0.00}
				: new double[]{0.00, 1.00};
  }

  /**
   * Make the node leaf node.
   * 
   * @param data 	training data
   */
  protected void makeLeaf(Instances data) {
    m_Attribute = null;
    m_isLeaf = true;
    m_ClassValue= data.firstInstance().value(data.numAttributes() - 1);
    m_ClassAttribute = data.classAttribute();
  }

  /**
   * Prints the decision tree using the protected toString method from below.
   * 
   * @return 		a textual description of the classifier
   */
  public String toString() {
	  if ((m_ClassProbs == null) && (m_Successors == null)) {
	      return "CART Tree: No model built yet.";
	    }

	    return "CART Decision Tree\n" + toString(0)+"\n\n"
	    +"Number of Leaf Nodes: "+numLeaves()+"\n\n" +
	    "Size of the Tree: "+numNodes();
  }

  /**
   * Outputs a tree at a certain level.
   * 
   * @param level 	the level at which the tree is to be printed
   * @return 		a tree at a certain level
   */
  protected String toString(int level) {

	  StringBuffer text = new StringBuffer();
	    // if leaf nodes
	    if (m_Attribute == null) {
	      if (Utils.isMissingValue(m_ClassValue)) {
		text.append(": null");
	      } else {
//		double correctNum = (int)(m_Distribution[Utils.maxIndex(m_Distribution)]*100)/
//		100.0;
//		double wrongNum = (int)((Utils.sum(m_Distribution) -
//		    m_Distribution[Utils.maxIndex(m_Distribution)])*100)/100.0;
//		String str = "("  + correctNum + "/" + wrongNum + ")";
//		text.append(": " + m_ClassAttribute.value((int) m_ClassValue)+ str);
	    	  text.append(": " + m_ClassAttribute.value((int) m_ClassValue));
	      }
	    } else {
	      for (int j = 0; j < 2; j++) {
		text.append("\n");
		for (int i = 0; i < level; i++) {
		  text.append("|  ");
		}
		if (j==0) {
		  if (m_Attribute.isNumeric())
		    text.append(m_Attribute.name() + " <= " + m_SplitValue);
		  else
		    text.append(m_Attribute.name() + "=" + m_SplitString);
		} else {
		  if (m_Attribute.isNumeric())
		    text.append(m_Attribute.name() + " > " + m_SplitValue);
		  else
		    text.append(m_Attribute.name() + "!=" + m_SplitString);
		}
			text.append(m_Successors[j].toString(level + 1));
	      }
	    }
	    return text.toString();
  }

  /**
   * Compute size of the tree.
   * 
   * @return 		size of the tree
   */
  public int numNodes() {
    if (m_isLeaf) {
      return 1;
    } else {
      int size =1;
      for (int i=0;i<m_Successors.length;i++) {
	size+=m_Successors[i].numNodes();
      }
      return size;
    }
  }

  /**
   * Method to count the number of inner nodes in the tree.
   * 
   * @return 		the number of inner nodes
   */
  public int numInnerNodes(){
    if (m_Attribute==null) return 0;
    int numNodes = 1;
    for (int i = 0; i < m_Successors.length; i++)
      numNodes += m_Successors[i].numInnerNodes();
    return numNodes;
  }

  /**
   * Return a list of all inner nodes in the tree.
   * 
   * @return 		the list of all inner nodes
   */
  protected Vector getInnerNodes(){
    Vector nodeList = new Vector();
    fillInnerNodes(nodeList);
    return nodeList;
  }

  /**
   * Fills a list with all inner nodes in the tree.
   * 
   * @param nodeList 	the list to be filled
   */
  protected void fillInnerNodes(Vector nodeList) {
    if (!m_isLeaf) {
      nodeList.add(this);
      for (int i = 0; i < m_Successors.length; i++)
	m_Successors[i].fillInnerNodes(nodeList);
    }
  }

  /**
   * Compute number of leaf nodes.
   * 
   * @return 		number of leaf nodes
   */
  public int numLeaves() {
    if (m_isLeaf) return 1;
    else {
      int size=0;
      for (int i=0;i<m_Successors.length;i++) {
	size+=m_Successors[i].numLeaves();
      }
      return size;
    }
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return 		an enumeration of all the available options.
   */
  public Enumeration listOptions() {
    Vector 	result;
    Enumeration	en;
    
    result = new Vector();
    
    en = super.listOptions();
    while (en.hasMoreElements())
      result.addElement(en.nextElement());

    result.addElement(new Option(
	"\tThe minimal number of instances at the terminal nodes.\n" 
	+ "\t(default 2)",
	"M", 1, "-M <min no>"));
    
    result.addElement(new Option(
	"\tThe number of folds used in the minimal cost-complexity pruning.\n"
	+ "\t(default 5)",
	"N", 1, "-N <num folds>"));
    
    result.addElement(new Option(
	"\tDon't use the minimal cost-complexity pruning.\n"
	+ "\t(default yes).",
	"U", 0, "-U"));
    
    result.addElement(new Option(
	"\tDon't use the heuristic method for binary split.\n"
	+ "\t(default true).",
	"H", 0, "-H"));
    
    result.addElement(new Option(
	"\tUse 1 SE rule to make pruning decision.\n"
	+ "\t(default no).",
	"A", 0, "-A"));
    
    result.addElement(new Option(
	"\tPercentage of training data size (0-1].\n" 
	+ "\t(default 1).",
	"C", 1, "-C"));

    return result.elements();
  }

  /**
   * Parses a given list of options. <p/>
   * 
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -S &lt;num&gt;
   *  Random number seed.
   *  (default 1)</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   * <pre> -M &lt;min no&gt;
   *  The minimal number of instances at the terminal nodes.
   *  (default 2)</pre>
   * 
   * <pre> -N &lt;num folds&gt;
   *  The number of folds used in the minimal cost-complexity pruning.
   *  (default 5)</pre>
   * 
   * <pre> -U
   *  Don't use the minimal cost-complexity pruning.
   *  (default yes).</pre>
   * 
   * <pre> -H
   *  Don't use the heuristic method for binary split.
   *  (default true).</pre>
   * 
   * <pre> -A
   *  Use 1 SE rule to make pruning decision.
   *  (default no).</pre>
   * 
   * <pre> -C
   *  Percentage of training data size (0-1].
   *  (default 1).</pre>
   * 
   <!-- options-end -->
   * 
   * @param options the list of options as an array of strings
   * @throws Exception if an options is not supported
   */
  public void setOptions(String[] options) throws Exception {
    String	tmpStr;
    
    super.setOptions(options);
    
    tmpStr = Utils.getOption('M', options);
    if (tmpStr.length() != 0)
      setMinNumObj(Double.parseDouble(tmpStr));
    else
      setMinNumObj(2);

    tmpStr = Utils.getOption('N', options);
    if (tmpStr.length()!=0)
      setNumFoldsPruning(Integer.parseInt(tmpStr));
    else
      setNumFoldsPruning(5);

    setUsePrune(!Utils.getFlag('U',options));
    setHeuristic(!Utils.getFlag('H',options));
    setUseOneSE(Utils.getFlag('A',options));

    tmpStr = Utils.getOption('C', options);
    if (tmpStr.length()!=0)
      setSizePer(Double.parseDouble(tmpStr));
    else
      setSizePer(1);

    Utils.checkForRemainingOptions(options);
  }

  /**
   * Gets the current settings of the classifier.
   * 
   * @return 		the current setting of the classifier
   */
  public String[] getOptions() {
    int       	i;
    Vector    	result;
    String[]  	options;

    result = new Vector();

    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);

    result.add("-M");
    result.add("" + getMinNumObj());
    
    result.add("-N");
    result.add("" + getNumFoldsPruning());
    
    if (!getUsePrune())
      result.add("-U");
    
    if (!getHeuristic())
      result.add("-H");
    
    if (getUseOneSE())
      result.add("-A");
    
    result.add("-C");
    result.add("" + getSizePer());

    return (String[]) result.toArray(new String[result.size()]);	  
  }

  /**
   * Return an enumeration of the measure names.
   * 
   * @return 		an enumeration of the measure names
   */
  public Enumeration enumerateMeasures() {
    Vector result = new Vector();
    
    result.addElement("measureTreeSize");
    
    return result.elements();
  }

  /**
   * Return number of tree size.
   * 
   * @return 		number of tree size
   */
  public double measureTreeSize() {
    return numNodes();
  }

  /**
   * Returns the value of the named measure.
   * 
   * @param additionalMeasureName 	the name of the measure to query for its value
   * @return 				the value of the named measure
   * @throws IllegalArgumentException 	if the named measure is not supported
   */
  public double getMeasure(String additionalMeasureName) {
    if (additionalMeasureName.compareToIgnoreCase("measureTreeSize") == 0) {
      return measureTreeSize();
    } else {
      throw new IllegalArgumentException(additionalMeasureName
	  + " not supported (Cart pruning)");
    }
  }

  /**
   * Returns the tip text for this property
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String minNumObjTipText() {
    return "The minimal number of observations at the terminal nodes (default 2).";
  }

  /**
   * Set minimal number of instances at the terminal nodes.
   * 
   * @param value 	minimal number of instances at the terminal nodes
   */
  public void setMinNumObj(double value) {
    m_minNumObj = value;
  }

  /**
   * Get minimal number of instances at the terminal nodes.
   * 
   * @return 		minimal number of instances at the terminal nodes
   */
  public double getMinNumObj() {
    return m_minNumObj;
  }

  /**
   * Returns the tip text for this property
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String numFoldsPruningTipText() {
    return "The number of folds in the internal cross-validation (default 5).";
  }

  /** 
   * Set number of folds in internal cross-validation.
   * 
   * @param value 	number of folds in internal cross-validation.
   */
  public void setNumFoldsPruning(int value) {
    m_numFoldsPruning = value;
  }

  /**
   * Set number of folds in internal cross-validation.
   * 
   * @return 		number of folds in internal cross-validation.
   */
  public int getNumFoldsPruning() {
    return m_numFoldsPruning;
  }

  /**
   * Return the tip text for this property
   * 
   * @return 		tip text for this property suitable for displaying in 
   * 			the explorer/experimenter gui.
   */
  public String usePruneTipText() {
    return "Use minimal cost-complexity pruning (default yes).";
  }

  /** 
   * Set if use minimal cost-complexity pruning.
   * 
   * @param value 	if use minimal cost-complexity pruning
   */
  public void setUsePrune(boolean value) {
    m_Prune = value;
  }

  /** 
   * Get if use minimal cost-complexity pruning.
   * 
   * @return 		if use minimal cost-complexity pruning
   */
  public boolean getUsePrune() {
    return m_Prune;
  }

  /**
   * Returns the tip text for this property
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui.
   */
  public String heuristicTipText() {
    return 
        "If heuristic search is used for binary split for nominal attributes "
      + "in multi-class problems (default yes).";
  }

  /**
   * Set if use heuristic search for nominal attributes in multi-class problems.
   * 
   * @param value 	if use heuristic search for nominal attributes in 
   * 			multi-class problems
   */
  public void setHeuristic(boolean value) {
    m_Heuristic = value;
  }

  /** 
   * Get if use heuristic search for nominal attributes in multi-class problems.
   * 
   * @return 		if use heuristic search for nominal attributes in 
   * 			multi-class problems
   */
  public boolean getHeuristic() {return m_Heuristic;}

  /**
   * Returns the tip text for this property
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui.
   */
  public String useOneSETipText() {
    return "Use the 1SE rule to make pruning decisoin.";
  }

  /** 
   * Set if use the 1SE rule to choose final model.
   * 
   * @param value 	if use the 1SE rule to choose final model
   */
  public void setUseOneSE(boolean value) {
    m_UseOneSE = value;
  }

  /**
   * Get if use the 1SE rule to choose final model.
   * 
   * @return 		if use the 1SE rule to choose final model
   */
  public boolean getUseOneSE() {
    return m_UseOneSE;
  }

  /**
   * Returns the tip text for this property
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui.
   */
  public String sizePerTipText() {
    return "The percentage of the training set size (0-1, 0 not included).";
  }

  /** 
   * Set training set size.
   * 
   * @param value 	training set size
   */  
  public void setSizePer(double value) {
    if ((value <= 0) || (value > 1))
      System.err.println(
	  "The percentage of the training set size must be in range 0 to 1 "
	  + "(0 not included) - ignored!");
    else
      m_SizePer = value;
  }

  /**
   * Get training set size.
   * 
   * @return 		training set size
   */
  public double getSizePer() {
    return m_SizePer;
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 8109 $");
  }
  
  
  public double cosSim(Instance first, Instance second, int numAttributes) {
		double sim = 0;
		double dotprod = 0;
		double norm1 = 0;
		double norm2 = 0;
		for (int k = 1; k<  numAttributes; k++ ) {
			dotprod += first.value(k) * second.value(k);
			if (first.isMissing(k) || second.isMissing(k)) {
//				System.out.println(dotprod);
				return -10.00;
			}
			norm1 += first.value(k) * first.value(k);
			norm2 += second.value(k) * second.value(k);
		}
		sim = dotprod/Math.sqrt(norm1*norm2);
		return sim;
	}

  /**
   * Main method.
   * @param args the options for the classifier
   */
  public static void main(String[] args) {
    runClassifier(new SimTree(), args);
  }
}
