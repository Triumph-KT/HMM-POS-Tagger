import java.io.*;
import java.util.*;
/**
 * A Bi-gram Hidden Markov Model (HMM) based Part-of-Speech (POS) Tagger.
 * This class implements training using tag counts and Viterbi decoding for tag sequence prediction.
 *
 * @author Triumph Kia Teh | Dartmouth CS10 | Winter 2025
 */
public class HMMTagger {

    //==================================================================
    // Instance Variables
    //==================================================================

    /** Stores tokenized sentences from training data (each sentence is a list of lowercase words) */
    protected List<List<String>> sentences;
    /** Stores corresponding POS tags for each sentence in training data */
    protected List<List<String>> tags;

    /** Transition counts between tags: Map<PreviousTag, Map<NextTag, Count>> */
    protected Map<String, Map<String, Integer>> transitionCounts;
    /** Observation counts of words per tag: Map<Tag, Map<Word, Count>> */
    protected Map<String, Map<String, Integer>> observationCounts;
    /** Total occurrences of each tag in training data */
    protected Map<String, Integer> tagCounts;

    // New maps to store log probabilities.
    /** Log probabilities for tag transitions: Map<PrevTag, Map<NextTag, LogProb>> */
    protected Map<String, Map<String, Double>> transitionLogProbs;
    /** Log probabilities for word observations: Map<Tag, Map<Word, LogProb>> */
    protected Map<String, Map<String, Double>> observationLogProbs;

    /** Log probability penalty (log probability) for unseen words/transitions */
    protected static final double UNSEEN_LOG_PROB = -10.0;

    /**
     * Constructor: Initializes data structures.
     */
    public HMMTagger() {
        sentences = new ArrayList<>();
        tags = new ArrayList<>();
        transitionCounts = new HashMap<>();
        observationCounts = new HashMap<>();
        tagCounts = new HashMap<>();

        transitionLogProbs = new HashMap<>();
        observationLogProbs = new HashMap<>();
    }

    //==================================================================
    // Core Methods
    //==================================================================

    /**
     * Reads sentences and corresponding POS tags from the training files.
     * @param sentencesFile The file containing sentences (one per line).
     * @param tagsFile The file containing POS tags (one per line, aligned with sentences).
     */
    public void loadTrainingData(String sentencesFile, String tagsFile) {
        try {
            BufferedReader sentenceReader = new BufferedReader(new FileReader(sentencesFile));
            BufferedReader tagReader = new BufferedReader(new FileReader(tagsFile));
            String sentenceLine, tagLine;

            // Read each line from both files simultaneously.
            while ((sentenceLine = sentenceReader.readLine()) != null &&
                    (tagLine = tagReader.readLine()) != null) {

                // Convert sentence to lowercase, trim spaces, and tokenize.
                List<String> sentenceTokens = new ArrayList<>();
                for (String word : sentenceLine.toLowerCase().trim().split("\\s+")) {
                    sentenceTokens.add(word);
                }

                // Tokenize the corresponding tag line.
                List<String> tagTokens = new ArrayList<>();
                for (String tag : tagLine.trim().split("\\s+")) {
                    tagTokens.add(tag);
                }

                // Save the processed data.
                sentences.add(sentenceTokens);
                tags.add(tagTokens);
            }

            sentenceReader.close();
            tagReader.close();
            System.out.println("Training data loaded successfully!");

        } catch (IOException e) {
            System.out.println("Error reading files: " + e.getMessage());
        }
    }

    /**
     * Trains the HMM by counting tag transitions and word observations.
     * After counting, it converts the counts to log probabilities.
     */
    public void trainHMM() {
        for (int i = 0; i < tags.size(); i++) {
            List<String> tagSequence = tags.get(i);      // POS tag sequence.
            List<String> wordSequence = sentences.get(i);  // Corresponding word sequence.

            if (tagSequence.isEmpty() || wordSequence.isEmpty()) continue; // Skip empty lines

            String firstTag = tagSequence.get(0); // First tag in the sentence

            // Ensure `#` is counted as a valid "tag" to avoid NullPointerException
            tagCounts.putIfAbsent("#", 0);
            tagCounts.put("#", tagCounts.get("#") + 1);

            // Handle start state `#` transition
            transitionCounts.putIfAbsent("#", new HashMap<>());
            transitionCounts.get("#").put(firstTag, transitionCounts.get("#").getOrDefault(firstTag, 0) + 1);

            for (int j = 0; j < tagSequence.size(); j++) {
                String currentTag = tagSequence.get(j);
                String currentWord = wordSequence.get(j);

                // Count the occurrence of the current tag.
                tagCounts.put(currentTag, tagCounts.getOrDefault(currentTag, 0) + 1);

                // Count the observation: word given tag.
                observationCounts.putIfAbsent(currentTag, new HashMap<>());
                observationCounts.get(currentTag).put(currentWord,
                        observationCounts.get(currentTag).getOrDefault(currentWord, 0) + 1);

                // Count the transition: previous tag to current tag.
                if (j > 0) { // Skip the first word (handled separately above)
                    String prevTag = tagSequence.get(j - 1);
                    transitionCounts.putIfAbsent(prevTag, new HashMap<>());
                    transitionCounts.get(prevTag).put(currentTag,
                            transitionCounts.get(prevTag).getOrDefault(currentTag, 0) + 1);
                }
            }
        }

        System.out.println("HMM training completed!");
        // After counting, compute the log probabilities.
        computeLogProbabilities();
    }

    /**
     * Converts raw counts to log probabilities.
     * For transitions: log P(tag2 | tag1) = log (count(tag1 → tag2) / total transitions from tag1)
     * For observations: log P(word | tag) = log (count(word, tag) / total occurrences of tag)
     */
    private void computeLogProbabilities() {
        // Compute log probabilities for tag transitions.
        for (String prevTag : transitionCounts.keySet()) {
            Map<String, Integer> innerMap = transitionCounts.get(prevTag);

            // sum up all transitions from prevTag
            int totalCount = 0;
            for (int count : innerMap.values()) {
                totalCount += count;  // Sum all outgoing transitions
            }

            // Avoid division by zero in case of errors in training data
            if (totalCount == 0) continue;

            Map<String, Double> logMap = new HashMap<>();
            for (String nextTag : innerMap.keySet()) {
                int count = innerMap.get(nextTag);
                double probability = (double) count / totalCount;
                double logProbability = Math.log(probability); // Natural log.
                logMap.put(nextTag, logProbability);
            }
            transitionLogProbs.put(prevTag, logMap);
        }

        // Compute log probabilities for observations (word given tag).
        for (String tag : observationCounts.keySet()) {
            Map<String, Integer> innerMap = observationCounts.get(tag);
            int totalCount = tagCounts.get(tag); // Total occurrences of tag.

            // Avoid division by zero
            if (totalCount == 0) continue;

            Map<String, Double> logMap = new HashMap<>();
            for (String word : innerMap.keySet()) {
                int count = innerMap.get(word);
                double probability = (double) count / totalCount;
                double logProbability = Math.log(probability); // Natural log.
                logMap.put(word, logProbability);
            }
            observationLogProbs.put(tag, logMap);
        }
    }

    /**
     * Performs Viterbi decoding to find the best sequence of tags for a given sentence.
     * @param sentenceStr The input sentence as a String.
     * @return The best sequence of tags as a List of Strings.
     */
    public List<String> viterbi(String sentenceStr) {
        // Tokenize the input sentence (convert to lowercase).
        String[] words = sentenceStr.toLowerCase().trim().split("\\s+");
        int n = words.length;

        // Get the set of all possible tags from training.
        Set<String> allTags = tagCounts.keySet();

        // dp[i] holds a map: tag -> best log probability score for word i.
        List<Map<String, Double>> dp = new ArrayList<>();
        // backPointers[i] holds a map: tag -> previous tag that led to best score at word i.
        List<Map<String, String>> backPointers = new ArrayList<>();

        // Initialization: for the first word, we assume a virtual start state with score 0.
        Map<String, Double> dp0 = new HashMap<>();
        Map<String, String> bp0 = new HashMap<>();
        for (String tag : allTags) {
            // Get observation log probability for the first word given this tag.
            // If word is unseen for the tag, use UNSEEN_LOG_PROB.
            double obsLogProb;
            if (observationLogProbs.containsKey(tag) && observationLogProbs.get(tag).containsKey(words[0])) {
                obsLogProb = observationLogProbs.get(tag).get(words[0]);
            } else {
                obsLogProb = UNSEEN_LOG_PROB;
            }
            double transLogProb = transitionLogProbs.getOrDefault("#", new HashMap<>()).getOrDefault(tag, UNSEEN_LOG_PROB);
            // Since we have no previous state, the initial score is just the observation probability.
            dp0.put(tag, transLogProb + obsLogProb);
            bp0.put(tag, null); // No backpointer for the first word.
        }
        dp.add(dp0);
        backPointers.add(bp0);

        // Fill in dp table for subsequent words 1 to n-1.
        for (int i = 1; i < n; i++) {
            Map<String, Double> dpi = new HashMap<>();
            Map<String, String> bpi = new HashMap<>();
            String word = words[i];

            // For each possible current tag.
            for (String currTag : allTags) {
                double bestScore = Double.NEGATIVE_INFINITY;
                String bestPrevTag = null;

                // Get observation log probability for current word given currTag.
                double obsLogProb;
                if (observationLogProbs.containsKey(currTag) && observationLogProbs.get(currTag).containsKey(word)) {
                    obsLogProb = observationLogProbs.get(currTag).get(word);
                } else {
                    obsLogProb = UNSEEN_LOG_PROB;
                }

                // For each possible previous tag (the best), compute candidate score.
                for (String prevTag : allTags) {
                    // Previous best score.
                    double prevScore = dp.get(i - 1).getOrDefault(prevTag, Double.NEGATIVE_INFINITY);

                    // Apply UNSEEN_LOG_PROB for missing transitions
                    double transLogProb = transitionLogProbs.getOrDefault(prevTag, new HashMap<>()).getOrDefault(currTag, UNSEEN_LOG_PROB);

                    // Candidate score for currTag at position i.
                    double candidateScore = prevScore + transLogProb + obsLogProb;

                    if (candidateScore > bestScore) {
                        bestScore = candidateScore;
                        bestPrevTag = prevTag;
                    }
                }
                dpi.put(currTag, bestScore);
                bpi.put(currTag, bestPrevTag);
            }
            dp.add(dpi);
            backPointers.add(bpi);
        }

        // Termination: find the best tag for the last word.
        double bestFinalScore = Double.NEGATIVE_INFINITY;
        String bestFinalTag = null;
        Map<String, Double> lastDp = dp.get(n - 1);
        for (String tag : lastDp.keySet()) {
            double score = lastDp.get(tag);
            if (score > bestFinalScore) {
                bestFinalScore = score;
                bestFinalTag = tag;
            }
        }

        // Backtrace to retrieve the best tag sequence (the best path).
        List<String> bestTagSequence = new ArrayList<>();
        String currTag = bestFinalTag;
        for (int i = n - 1; i >= 0; i--) {
            bestTagSequence.add(currTag);
            currTag = backPointers.get(i).get(currTag);
        }
        // The sequence is built backwards, so reverse it.
        Collections.reverse(bestTagSequence);
        return bestTagSequence;
    }

    //==================================================================
    // Testing/Utility Methods
    //==================================================================

    /**
     * Evaluates performance on test files by comparing predicted tags with gold-standard tags.
     * @param testSentencesFile Path to test sentences file.
     * @param testTagsFile Path to test tags file.
     */
    public void testPerformance(String testSentencesFile, String testTagsFile) {
        int totalTags = 0;
        int correctTags = 0;

        try {
            BufferedReader sentenceReader = new BufferedReader(new FileReader(testSentencesFile));
            BufferedReader tagReader = new BufferedReader(new FileReader(testTagsFile));
            String sentenceLine, tagLine;

            while ((sentenceLine = sentenceReader.readLine()) != null &&
                    (tagLine = tagReader.readLine()) != null) {
                String[] goldTags = tagLine.trim().split("\\s+");
                List<String> predictedTags = viterbi(sentenceLine);
                int len = Math.min(goldTags.length, predictedTags.size());
                for (int i = 0; i < len; i++) {
                    totalTags++;
                    if (goldTags[i].equals(predictedTags.get(i))) {
                        correctTags++;
                    }
                }
            }
            sentenceReader.close();
            tagReader.close();

            double accuracy;
            if (totalTags > 0) {accuracy = (double) correctTags / totalTags * 100;}
            else {accuracy = 0;}
            System.out.println("Total tags evaluated: " + totalTags);
            System.out.println("Correct tags: " + correctTags);
            System.out.printf("Accuracy: %.2f%%\n", accuracy);

        } catch (IOException e) {
            System.out.println("Error reading test files: " + e.getMessage());
        }
    }

    /**
     * Runs file-based tests on all provided datasets (example, simple, and brown).
     */
    public static void runFileBasedTests() {
        // Define the input directory.
        String inputDir = "inputs/";

        // --- Example Dataset ---
        System.out.println("\n=== Testing on Example Dataset ===");
        HMMTagger exampleTagger = new HMMTagger();
        // Using the example files for both training and testing (if separate test files are not provided).
        exampleTagger.loadTrainingData(inputDir + "example-sentences.txt", inputDir + "example-tags.txt");
        exampleTagger.trainHMM();
        exampleTagger.testPerformance(inputDir + "example-sentences.txt", inputDir + "example-tags.txt");

        // --- Simple Dataset ---
        System.out.println("\n=== Testing on Simple Dataset ===");
        HMMTagger simpleTagger = new HMMTagger();
        simpleTagger.loadTrainingData(inputDir + "simple-train-sentences.txt", inputDir + "simple-train-tags.txt");
        simpleTagger.trainHMM();
        simpleTagger.testPerformance(inputDir + "simple-test-sentences.txt", inputDir + "simple-test-tags.txt");

        // --- Brown Dataset ---
        System.out.println("\n=== Testing on Brown Dataset ===");
        HMMTagger brownTagger = new HMMTagger();
        brownTagger.loadTrainingData(inputDir + "brown-train-sentences.txt", inputDir + "brown-train-tags.txt");
        brownTagger.trainHMM();
        brownTagger.testPerformance(inputDir + "brown-test-sentences.txt", inputDir + "brown-test-tags.txt");
    }

    /**
     * Runs a console-based test allowing user input to get the corresponding tags.
     */
    public void runConsoleTest() {
        Scanner scanner = new Scanner(System.in);
        System.out.println("\nEnter a sentence to tag (type 'exit' to quit):");
        while (true) {
            System.out.print("Input: ");
            String inputLine = scanner.nextLine().trim();
            if (inputLine.equalsIgnoreCase("exit")) {
                System.out.println("Exiting console test.");
                break;
            }
            if (inputLine.isEmpty()) {
                System.out.println("Empty input. Please enter a valid sentence.");
                continue;
            }
            List<String> predictedTags = viterbi(inputLine);
            System.out.println("Predicted Tags: " + predictedTags);
        }
        scanner.close();
    }

    /**
     * Displays raw transition probabilities P(tag2 | tag1) (for debugging/comparison).
     */
    public void displayTransitionProbabilities() {
        System.out.println("\n Transition Probabilities (P(tag2 | tag1)):");
        for (String prevTag : transitionCounts.keySet()) {
            System.out.print(prevTag + " → ");
            for (String nextTag : transitionCounts.get(prevTag).keySet()) {
                int count = transitionCounts.get(prevTag).get(nextTag);
                int total = tagCounts.get(prevTag);
                double probability = (double) count / total;
                System.out.printf("%s (%.2f)  ", nextTag, probability);
            }
            System.out.println();
        }
    }

    /**
     * Displays raw observation probabilities P(word | tag) (for debugging/comparison).
     */
    public void displayObservationProbabilities() {
        System.out.println("\n Observation Probabilities (P(word | tag)):");
        for (String tag : observationCounts.keySet()) {
            System.out.print(tag + " → ");
            for (String word : observationCounts.get(tag).keySet()) {
                int count = observationCounts.get(tag).get(word);
                int total = tagCounts.get(tag);
                double probability = (double) count / total;
                System.out.printf("%s (%.2f)  ", word, probability);
            }
            System.out.println();
        }
    }

    /**
     * Displays computed log transition probabilities: log P(tag2 | tag1) (for debugging/comparison).
     */
    public void displayTransitionLogProbabilities() {
        System.out.println("\n Transition Log Probabilities (log P(tag2 | tag1)):");
        for (String prevTag : transitionLogProbs.keySet()) {
            System.out.print(prevTag + " → ");
            for (String nextTag : transitionLogProbs.get(prevTag).keySet()) {
                double logProb = transitionLogProbs.get(prevTag).get(nextTag);
                System.out.printf("%s (%.2f)  ", nextTag, logProb);
            }
            System.out.println();
        }
    }

    /**
     * Displays computed log observation probabilities: log P(word | tag) (for debugging/comparison).
     */
    public void displayObservationLogProbabilities() {
        System.out.println("\n Observation Log Probabilities (log P(word | tag)):");
        for (String tag : observationLogProbs.keySet()) {
            System.out.print(tag + " → ");
            for (String word : observationLogProbs.get(tag).keySet()) {
                double logProb = observationLogProbs.get(tag).get(word);
                System.out.printf("%s (%.2f)  ", word, logProb);
            }
            System.out.println();
        }
    }

    /**
     * Main method to run the program and test log probability conversion.
     */
    public static void main(String[] args) {
        // ----------------------------------------------------------------
        // 1. DEFAULT SETUP
        // ----------------------------------------------------------------
        // Directory where input files are stored:
        String inputDir = "inputs/";
        // File names for the SIMPLE training set:
        String sentencesFile = inputDir + "brown-train-sentences.txt";
        String tagsFile      = inputDir + "brown-train-tags.txt";

        // Initialize HMMTagger and train on the simple dataset:
        HMMTagger tagger = new HMMTagger();
        tagger.loadTrainingData(sentencesFile, tagsFile);
        tagger.trainHMM();

        // ----------------------------------------------------------------
        // 2. CONSOLE-BASED TEST
        // ----------------------------------------------------------------
        // Uncomment to allow interactive console tagging:
        tagger.runConsoleTest();

        // ----------------------------------------------------------------
        // 3. FILE-BASED TESTS FOR "example", "simple", "brown"
        // ----------------------------------------------------------------
        // Uncomment to run the helper method below, which trains on
        // each dataset and tests on it:
        // runFileBasedTests();

        // ----------------------------------------------------------------
        // 4. OPTIONAL: DEBUGGING / INSPECTION METHODS
        // ----------------------------------------------------------------
        // Uncomment any of the following to display raw or log probabilities:
//         tagger.displayTransitionProbabilities();
//         tagger.displayObservationProbabilities();
//
//         tagger.displayTransitionLogProbabilities();
//         tagger.displayObservationLogProbabilities();

        // ----------------------------------------------------------------
        // 5. OPTIONAL: CUSTOM TEST SENTENCES FOR MANUAL CHECKS
        // ----------------------------------------------------------------
//         String[] testSentences = {
//             "he trains the dog",
//             "your work is beautiful",
//             "trains are fast",
//             "we should watch the dog work in a cave",
//             "she sings a song",
//             "trains",
//             "unknown unknown",
//             "you work for unknown",
//             "You are very loving to me ."
//         };
//
//         System.out.println("\nViterbi Decoding Tests (Multiple Cases):");
//         for (String testSentence : testSentences) {
//             if (testSentence.trim().isEmpty()) {
//                 System.out.println("Input Sentence is empty. Skipping.");
//                 continue;
//             }
//             List<String> predictedTags = tagger.viterbi(testSentence);
//             System.out.println("Input Sentence: " + testSentence);
//             System.out.println("Predicted Tags: " + predictedTags);
//             System.out.println("-----------------------------");
//         }
    }
}
