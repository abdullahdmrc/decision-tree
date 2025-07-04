import java.io.*;
import java.util.*;

public class Main {

    private static HashMap<String,Integer> labelFrequencies;         // the last columns frequencies for calculating the entropy
    private static String[] featuresArray;      // keeping the features names
    private static  int totalLabelCounter=0;
    private static List<String[]> dataSet;      // all rows

    public static void main(String[] args) {
        labelFrequencies=new HashMap<>();
        dataSet = new ArrayList<>();
        Scanner scanner=new Scanner(System.in);
        System.out.println("Enter the file name: ");
        String filePath=scanner.nextLine();
        try {
            File Obj = new File(filePath);
            Scanner Reader = new Scanner(Obj);
            String features=Reader.nextLine();
            featuresArray=features.split("[,;.\\t]");   // splitting the file with different delimiter types
            while (Reader.hasNextLine()) {

                String data = Reader.nextLine().toLowerCase();
                String[] splittedData=data.split("[,;.\\t]");
                dataSet.add(splittedData);

                Integer labelFrequency=labelFrequencies.get(splittedData[splittedData.length-1]);
                if (labelFrequency==null){
                    labelFrequencies.put(splittedData[splittedData.length-1],1);
                }else {
                    labelFrequency++;
                    labelFrequencies.put(splittedData[splittedData.length-1],labelFrequency);
                }

                totalLabelCounter++;
            }
            Reader.close();
        }
        catch (FileNotFoundException e) {
            System.out.println("File not found, check the file name or path");
            //e.printStackTrace();
        }

        System.out.println("\n--- Initial Dataset Analysis ---");
        System.out.println("Label Frequencies: " + labelFrequencies);
        System.out.println("Total Samples: " + totalLabelCounter);


        List<String> featureList = new ArrayList<>(Arrays.asList(featuresArray));
        featureList.remove(featuresArray.length - 1);

        TreeNode<String> decisionTree = buildTree(dataSet, featureList);
        System.out.println("\n--- Decision Tree Structure ---");
        printTree(decisionTree, "", "");


        while (true) {                                              // user input interactions begin here
            Scanner inputScanner = new Scanner(System.in);
            Map<String, String> userInput = new HashMap<>(); // Initialize a map to store feature-value pairs entered by the user
            boolean exit = false;
            System.out.println("\n--Enter the values (Press Enter for exiting)--");

            for (int i = 0; i < featuresArray.length - 1; i++) {
                System.out.print(featuresArray[i] + ": ");
                String value = inputScanner.nextLine().trim().toLowerCase(); // Read user input, trim whitespace, convert to lowercase

                if (value.isEmpty()) {          // if the user enters nothing the program ends
                    exit = true;
                    break;
                }
                userInput.put(featuresArray[i], value);
            }
            if (exit) {
                System.out.println("Exiting...");
                break;
            }
            String prediction = predict(decisionTree, userInput);
            System.out.println("Prediction: " + prediction);
        }
    }
    /*
    * Calculates the entropy of a given dataset subset.
    * Steps:
    * Count how many times each label (class) appears in the subset.
    * Calculate the probability of each label.
    * Use the entropy formula:
    * Entropy(S) = - Σ (p_i * log2(p_i))
    * where p_i is the probability of label i.
    * */
    public static double calculateEntropyOfDataSet(List<String[]> data) { // this method for calculating entropy of the subsets
        int labelIndex = featuresArray.length - 1;
        Map<String, Integer> currentLabelFrequencies = new HashMap<>();// Map to count occurrences of each label
        int currentTotalSamples = 0;
        for (String[] row : data) {
            String label = row[labelIndex];
            currentLabelFrequencies.put(label, currentLabelFrequencies.getOrDefault(label, 0) + 1);
            currentTotalSamples++;
        }
        double entropy = 0;
        for (int count : currentLabelFrequencies.values()) {
            double p = (double) count / currentTotalSamples;// Probability of current label
            if (p > 0) {
                double log2 = Math.log(p) / Math.log(2);// Calculate log base 2 of p
                entropy += p * log2;    // Add weighted log probability to entropy sum
            }
        }
        return -entropy;
    }
    /*
     * Calculates the Information Gain for each feature in the given dataset.
     * Steps:
     * Calculate the total entropy of the dataset.
     * For each feature:
     * a. Partition the dataset into subsets based on feature values.
     * b. Compute weighted entropy of all subsets.
     * c. Subtract weighted entropy from total entropy to get Information Gain.
     */
    public static HashMap<String, Double> calculateInformationGainsForAllFeatures(List<String[]> dataSet) {  // Calculates the information gain for each feature in the given dataset.
        double datasetEntropy = calculateEntropyOfDataSet(dataSet);
        HashMap<String, Double> informationGains = new HashMap<>();
        int labelIndex = featuresArray.length - 1;                    // The information gain is calculated as the total entropy minus the weighted entropy of the subsets divided by that feature.

        for (int featureIndex = 0; featureIndex < labelIndex; featureIndex++) {
            String featureName = featuresArray[featureIndex];
            Map<String, List<String[]>> subsets = new HashMap<>();

            for (String[] row : dataSet) {    // Subset the dataset according to each value of the property
                String featureValue = row[featureIndex];
                subsets.putIfAbsent(featureValue, new ArrayList<>());//features and their rows
                subsets.get(featureValue).add(row);
            }
            double featureEntropy = 0.0;

            int currentDataSetSize = dataSet.size();
            // Calculate the entropy for each of the subsets and sum them weighted
            for (Map.Entry<String, List<String[]>> entry : subsets.entrySet()) {
                List<String[]> subset = entry.getValue();
                int subsetSize = subset.size();
                Map<String, Integer> labelCounts = new HashMap<>();
                for (String[] row : subset) {
                    String label = row[labelIndex];
                    labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
                }
                double subsetEntropy = 0.0;
                for (int count : labelCounts.values()) {
                    double p = (double) count / subsetSize;
                    if (p > 0) {
                        subsetEntropy += -p * (Math.log(p) / Math.log(2));
                    }
                }
                featureEntropy += ((double) subsetSize / currentDataSetSize) * subsetEntropy;
            }
            double infoGain = datasetEntropy - featureEntropy;
            informationGains.put(featureName, infoGain);
        }
        return informationGains;          // Returns a map of features and their respective information gains
    }
    /*
     * This method selects the best feature (based on Information Gain)
     * to split the dataset at the current level. It then creates a tree node
     * for this feature and branches for each unique value of the feature,
     * continuing the process recursively.
     * Base Cases:
     * If all labels in the current subset are the same, return a leaf node.
     * If there are no remaining features to split on, return a leaf node
     * with the majority label.
     * Recursive Case:
     * Calculate information gain for each remaining feature.
     * Select the feature with the highest information gain.
     * Partition the dataset by that feature's values.
     * Recursively build child nodes for each partition.
     */
    public static TreeNode<String> buildTree(List<String[]> data, List<String> remainingFeatures) {
        int labelIndex = featuresArray.length - 1;
        Set<String> uniqueLabels = new HashSet<>();    // Collect all unique labels in the current data subset
        for (String[] row : data) {
            uniqueLabels.add(row[labelIndex]);
        }
        String indent = getIndentString(data.size());
        System.out.println("\n" + indent + "--- Building Node (Current Data Subset Size: " + data.size() + ") ---");
        System.out.println(indent + "  Labels in current subset: " + getLabelFrequencies(data));
        double currentEntropy = calculateEntropyOfDataSet(data);
        System.out.println(indent + "  Entropy of current subset: " + String.format("%.4f", currentEntropy));

        // If all labels in the subset are the same, create a leaf node and return the label value
        if (uniqueLabels.size() == 1) {
            System.out.println(indent + "  Leaf Node (All labels are same): " + data.get(0)[labelIndex]);
            return new TreeNode<>(data.get(0)[labelIndex]); // Leaf
        }
        if (remainingFeatures.isEmpty()) {
            Map<String, Integer> labelCounts = new HashMap<>();
            for (String[] row : data) {
                String label = row[labelIndex];
                labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
            }
            String label = Collections.max(labelCounts.entrySet(), Map.Entry.comparingByValue()).getKey();
            System.out.println(indent + "  Leaf Node (No remaining features, most-frequent Label): " + label);
            return new TreeNode<>(label);
        }

        Map<String, Double> infoGains = calculateInformationGainsForAllFeatures(data);

        System.out.println(indent + "  Information Gains for remaining features:");
        String bestFeature = null;
        double maxGain = Double.NEGATIVE_INFINITY;
                                                            // Find the property with the highest information gain
        for (String feature : remainingFeatures) {
            double gain = infoGains.getOrDefault(feature, 0.0);
            System.out.println(indent + "    " + feature + ": " + String.format("%.4f", gain));
            if (gain > maxGain) {
                maxGain = gain;
                bestFeature = feature;
            }
        }
        System.out.println(indent + "  Best Feature selected: " + bestFeature + " (Gain: " + String.format("%.4f", maxGain) + ")");

        TreeNode<String> root = new TreeNode<>(bestFeature, null); // The selected feature is assigned as the root node
        int bestFeatureIndex = Arrays.asList(featuresArray).indexOf(bestFeature);    // Find index of the best feature in the features array
        Map<String, List<String[]>> partitions = new HashMap<>();  // Subset the data according to the values of the selected feature
        for (String[] row : data) {
            String value = row[bestFeatureIndex];
            partitions.putIfAbsent(value, new ArrayList<>());
            partitions.get(value).add(row);
        }
        List<String> updatedFeatures = new ArrayList<>(remainingFeatures);
        updatedFeatures.remove(bestFeature);

        for (Map.Entry<String, List<String[]>> entry : partitions.entrySet()) {// Branch for each subset
            String value = entry.getKey();
            List<String[]> subset = entry.getValue();

            System.out.println(indent + "  --> Branching on " + bestFeature + " = " + value);
            if (!subset.isEmpty()) {
                TreeNode<String> child = buildTree(subset, updatedFeatures);
                root.addChild(value, child);
            }
        }
        return root; // Return the constructed tree
    }

    /*
     * Counts the frequency of each class label in the given dataset.
     * Iterates over each row and extracts the class label (last column),
     * incrementing a counter for each label found.
     */
    private static Map<String, Integer> getLabelFrequencies(List<String[]> data) {
        int labelIndex = featuresArray.length - 1;
        Map<String, Integer> counts = new HashMap<>();
        for (String[] row : data) {
            String label = row[labelIndex];
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }
        return counts;
    }
    /*
     * Generates a string used for indentation in console output,
     * based on the depth of the current node in the decision tree.
     * The depth is estimated using the ratio of the original dataset size
     * to the current subset size, and applying log base 2.
     * This helps visualize the tree structure in a readable, indented format.
     */
    private static String getIndentString(int currentSubsetSize) {

        int totalOriginalSize = totalLabelCounter;
        int depth = 0;
        if (totalOriginalSize > 0) {    // Calculate the depth of the current node based on the size of the current subset compared to the original dataset size
            depth = (int) (Math.log((double)totalOriginalSize / currentSubsetSize) / Math.log(2));
        }
        StringBuilder sb = new StringBuilder();// Build a string of spaces for indentation
        for (int i = 0; i < depth; i++) {
            sb.append("  ");
        }
        return sb.toString();// Return the generated indentation string
    }
    /*
     * Recursively prints the structure of the decision tree using ASCII tree branches.
     * Each node is printed with appropriate indentation and branch symbols (├──, └──),
     * giving a clear visual representation of the tree hierarchy.
     * Leaf nodes are printed with their label values.
     * Internal nodes are printed with the feature name and the corresponding feature value.
     */
   public static void printTree(TreeNode<String> node, String prefix, String childPrefix) {
       if (node.isLeaf()) {
           System.out.println(prefix + "Label: " + node.getLabel());
           return;
       }

       List<Map.Entry<String, TreeNode<String>>> childrenList = new ArrayList<>(node.getChildren().entrySet());
       for (int i = 0; i < childrenList.size(); i++) {
           Map.Entry<String, TreeNode<String>> entry = childrenList.get(i);
           boolean isLast = (i == childrenList.size() - 1);

           System.out.println(prefix + (isLast ? "└── " : "├── ") + node.getFeature() + " = " + entry.getKey());
           printTree(entry.getValue(), childPrefix + (isLast ? "    " : "│   "), childPrefix + (isLast ? "    " : "│   "));
       }
   }
    /*
     * Predicts the class label for a given input using the trained decision tree.
     * This method first checks if the input exactly matches any training sample:
     * If a unique matching label is found, that label is returned.
     * If conflicting labels are found for the same input, returns "Undecidable".
     * If no exact match exists, it traverses the decision tree from the root node:
     * At each decision node, it selects the child branch corresponding to the input value.
     * If a required feature value is missing or not present in the tree, it returns "Unknown".
     * If a leaf node is reached, it returns the associated label.
     */
    public static String predict(TreeNode<String> root, Map<String, String> input) {

        int labelIndex = featuresArray.length - 1;
        Set<String> possibleLabels = new HashSet<>();
        boolean exactMatchFoundInTraining = false;

        for (String[] row : dataSet) {// First, check if the input matches exactly any training data row
            boolean featuresMatch = true;
            for (int i = 0; i < labelIndex; i++) {
                String featureName = featuresArray[i];

                if (!input.containsKey(featureName) || !input.get(featureName).equals(row[i])) {// If input value for feature is missing or different, not a match
                    featuresMatch = false;
                    break;
                }
            }
            if (featuresMatch) {  // If exact match found, collect corresponding labels
                possibleLabels.add(row[labelIndex]);
                exactMatchFoundInTraining = true;
            }
        }

        if (exactMatchFoundInTraining && possibleLabels.size() > 1) {    // If multiple different labels found for exact same input in training, prediction is undecidable
            return "Undecidable (conflicting data in training set for this input)";
        }

        TreeNode<String> currentNode = root;
        while (!currentNode.isLeaf()) {    // Traverse the decision tree from root until reaching a leaf node
            String feature = currentNode.getFeature();
            String value = input.get(feature);

            if (value == null || !currentNode.getChildren().containsKey(value)) {        // If input missing value for this feature or branch doesn't exist in tree, prediction unknown
                return "Unknown (value not found in tree)";
            }
            currentNode = currentNode.getChildren().get(value);
        }
        return currentNode.getLabel();
    }

    static class TreeNode<T> {
        private String feature;                  // outlook etc.
        private T value;                         // sunny etc
        private T label;                         // yer or no etc
        private Map<T, TreeNode<T>> children;
        // Constructor for internal node
        public TreeNode(String feature, T value) {
            this.feature = feature;
            this.value = value;
            this.label = null;
            this.children = new HashMap<>();
        }
        // Constructor for leaf node
        public TreeNode(T label) {
            this.feature = null;
            this.value = null;
            this.label = label;
            this.children = new HashMap<>();
        }
        public boolean isLeaf() {
            return label != null;
        }
        public void addChild(T value, TreeNode<T> child) {
            children.put(value, child);
        }
        // Getter methods
        public String getFeature() {
            return feature;
        }
        public T getValue() {
            return value;
        }
        public T getLabel() {
            return label;
        }
        public Map<T, TreeNode<T>> getChildren() {
            return children;
        }
    }
}