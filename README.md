# HMM-POS-Tagger
Java-based Part-of-Speech Tagger using Hidden Markov Models and Viterbi Algorithm.

## Overview
A Java-based Part-of-Speech (POS) tagger utilizing **Hidden Markov Models (HMM)** and the **Viterbi Algorithm** to accurately tag words in a sentence. Achieves **96.5% accuracy** on the Brown corpus.

## Technologies
- Java
- NLP (Natural Language Processing)
- Hidden Markov Models
- Viterbi Algorithm

## Features
- Trained on the Brown corpus.
- Computes POS transition and emission probabilities.
- Applies smoothing for unseen words (OOV handling).
- Efficient backtracking to optimize inference speed.

## How to Run
1. Compile the Java files:
   ```bash
   javac src/*.java
