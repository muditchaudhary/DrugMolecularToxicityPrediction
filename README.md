# DrugMolecularToxicityPrediction

## Project Specification
Almost all people are exposed to different chemicals during their lifetimes through different
sources including food, household cleaning products and medicines. However, in some cases,
these chemicals can be toxic and affect human health. As a matter of fact, over 30% of drugs have
failed in human clinical trials because they are determined to be toxic despite promising preclinical
studies in animal models. Consider real-world clinical trials for assessing drugs are
extremely time-consuming, it is ideal if a computational drug molecular toxicity assessment
method can be developed to quickly test whether certain chemicals have the potential to disrupt
processes in the human body of great concern to human health.

Deep neural network has become a hot research topic in machine learning in recent years.
Compared to other methods, deep learning has shown its advantages in handling large amount of
data and achieving better performance. In this project, we have a dataset of
drug molecules with their SMILES expressions (which will be explained later) and the binary
labels indicating whether one drug molecule is toxic or not.

## SMILES Expression
Simplified Molecular-Input Line-Entry System (SMILES) is a linear representation for
molecular structure using 1D ASCII strings. For example, aspirin, a commonly used drug in daily
life, its SMILES is CC(=O)OC1=CC=CC=C1C(=O)O

The one hot format of SMILES is a 2D {0,1} matrix, where each column represents a symbol in
the SMILES notation of the current molecule, and each row is one ASCII character appeared in the
dataset’s SMILES dictionary. The size of the 2D matrices is the size of the dataset’s SMILES
dictionary * the length of the longest molecule SMILES, which means we have zeros padded after
short molecule SMILES. For a SMILES notation, one at row i, col j means the jth symbol of that
SMILES is the ith character in the dictionary.

## Dataset
The dataset provided is about the toxicity of some small molecules. We provide two folders for
you, one is the training data NR-ER-train (about 8k samples), and the other one is testing data NR-ER-test
(about hundreds of samples). There are three files in each folder:
    * names_smiles.csv: (String): A csv file, each line contains a drug molecule’s name and its SMILES expression, separated by comma (,)
    * names_labels.csv: (String): A csv file, each line contains a drug molecule’s name and its toxicity label, where 0 means nontoxic and 1 means toxic, separated by comma (,)
    * names_onehots.npy: (Numeric): An npy file which can be loaded by numpy package, storing two ndarray; one is the names of the molecules, and the other is the one-hot representations of SMILES expressions of drug molecules. Unzip the file for NR-ER-train.

## Code
* train.py: Use this to train your data. Change the address values according to your system.
* test.py: Use this to test your data and predict the values in a labels.txt file.
