# CS 4740: Intro to NLP

# Project 2: Named Entity Recognition

# with HMMs and MEMMs

#### Jonathan Ho (jh964), Daniel Goldfarb (dg393), Junan Qu (jq77)

#### Kaggle Team Name: QuQuTrain


## Preprocessing

After examining the dataset given and the brief introduction on the handout, we decided to pre-process
the training dataset by integrating every three elements into one unit instance. For example, the following
is an example from train.txt:
_Figure 1: Dataset Snapshot_
Each sentence is associated with the corresponding Part-Of-Speech tags and labels. 
We then converted the string sentences into lists so we can easily iterate through by token using the ​ split_raw_data ​ function.

This function maintains the sentence structure so that the beginning of each training example of
independent from the previous one. Lastly, we decided to convert every word to all lowercase letters. We
made this decision because we noticed that many times capitalized words from the beginning of the
sentence were classified drastically differently than their lowercase counterpart.

## HMM Implementation

### Baseline

We first developed a baseline system that served as a model that our HMM could be compared to. We
followed the design of the ​ **most frequent class baseline** ​​, that always assigns each token to the class it
occurred in most often in the training set.
We wrote a method that traverses through the entire training dataset and builds a dictionary that takes
every word token as the key and its corresponding tags as the values. From this dictionary, we then
calculate the ​ **counts** ​​ of every tag with respect to every word. This essentially creates a ​ **probability
distribution** ​​ for the possible tags that could be assigned to a word based on the frequency in the training
corpus. We can then sample this distribution by picking a label from the set uniformly at random.
The HMM implementation scored ​ **67% on the Kaggle** ​​ competition test set.

### Start Probability

To calculate each tag’s probability of being the ​ **starting tag** ​​, we first need to calculate the counts of each
tag that is seen to be the starting tag. Therefore, we decided to iterate through the training corpus and
build a dictionary that has every possible tag as the keys. Whenever a tag is seen to be a starting tag for
a sequence, we increase the count of tag by one. We then calculate the probability by dividing the
number of counts by the total number of starting tags. Eventually, we have a dictionary that every key is a
tag, and its value is the probability of that tag being the starting tag. This value is multiplied with the
emission probability to fill the first stage of the Viterbi algorithm.


_Figure 2: Starting Probability Chart_
Figure 2 shows the result of starting probability we got from training data. We used a plus-1 smoothing
technique for these probabilities so that explains why the I-XXX tags have a nonzero probability. Given
that the number of counts is equal to the size of the training corpus, we decided that a value of 1 was
arbitrarily small enough to not disturb the probability distribution while still assigning a non-zero probability
to even the least likely scenarios.

### Transition Probability

#### P ( t ) ≈Π i ≥ 1 PT ( ti | ti − 1 )

**We are building a conditional model that gives the probability of generating the current tag given
the previous tag.** ​​We implemented this through a “double-dictionary”. We chose this data structure as it
is more elegant to search for the probabilities using the actual word tokens themselves rather than integer
indices. The keys for both layers are each of the 9 possible tags for word tokens. The function get_transition_probs ​() returns this dictionary which we assign to the dict ​ trans_probabilities ​. So calling trans_probabilities ​[t_1][t_2] gives us the probability of the current word having tag t_1 given that the

previous word had tag t_2. Some of these transition probabilities turned out to be 0 after training on the
corpus. For example, we never saw a B-MISC tag following a B-PER tag in the training corpus but this
scenario is certainly possible in the wild. So we applied plus-1 smoothing to distribute a small portion of the probability mass to these scenarios. The ​ k ​ value of 1 was chosen for the same reason as for the start probabilities.
The following chart is a visualization of the structure we designed. Figure 3 is an example of the bigram of
P(t _i_ |​t _i_ − 1 ) where t _i_ −^1 is B-LOC in this case. Intuitively, these values make sense. B-LOC following a
B-LOC is pretty unlikely while B-LOC following I-LOC is very likely.
_Figure 3: Bigram Model Double-Dictionary - B-LOC Entry_


### Emission Probability

_P_ ( _t_ ) ≈Π _i_ ≥ 1 _P_ ( _wi_ | _ti_ )
**Given a tag, we are looking for the probability of generating a certain word.** ​​We made the choice to
use the “double-dictionary” design just like in our calculation of the transition probabilities. Similarly, these
probabilities were calculated by counting each of the gold standard tags that correspond to each word
token in the training corpus.
Here we have the possibility of encountering unseen bigrams. This would occur when we come across a
combination of a word given a tag, t, where the word has been seen before but never with the specific tag
t. We solved this problem by applying plus-k smoothing. By splitting the training corpus into a 70/
train/validation split, we applied a grid search hyperparameter optimization technique to find an optimal ​ _k_
value. This value turned out to be 0.01 (grid search accuracies shown in Figure 4).
It is also possible to run into the problem of an unknown word in the test set that we do not have a
specified dictionary for from the training corpus. We decided to assign a probability distribution for the
unknown words based on the lowest probability, P*(w|t), of the word given a tag (our reasoning for using
this technique is described in the Experiments section). For each tag, the unknown probability is assigned
probability q(P*(w|t)) for some constant q. Similar to finding the optimal ​ k ​ for smoothing, we found this ​ q
value through a grid search and ended up using 1/6 (grid search accuracies shown in Figure 4).

### Viterbi Algorithm

In our implementation of Viterbi, we maintained two double-dictionaries with the following invariants: the
first maintains the probability of the best path to a given index with a given tag and the second maintains
the path that generated the above probability. We fill out these dictionaries word by word, starting from
index = 0, and using only the previous index’s probabilities and paths to compute the next set of
probabilities and paths. Theoretically, we could save memory by implementing only two columns
representing the previous index and the index currently being calculated for each state, but we figured
that the memory savings were small, and being able to investigate how the paths were built up is a
valuable tool for both debugging and parameter-adjusting.

### Experiments

#### Expectation/Hypothesis: ​ We expect the HMM to outperform our baseline system because we are now

incorporating conditional probabilities that use information based on the previous word that our baseline
did not have access to.
All of our proposed models were experimented by splitting the training corpus into a 70/30 train/validation
split. By doing this, we are able to compare accuracies when experimenting on a held-out set from the
corpus that we tested on and deciding on the model variation that minimizes this validation accuracy.
The two main hyper-parameters that we were experimenting with were ​ k ​ which was our laplacian
smoothing parameter and ​ q ​ which denoted how much to penalize unknown word probabilities in the

distribution. Our grid search comparisons can be seen in Figure 4.


For unknown word handling, our initial technique to handle these was to assume that all words that only
occur once are unknown. Although this technique worked well for the corpus in project 1, it performed
poorly for us in named entity recognition. This is because many rare named entities in the training corpus
only appeared once, but these word types are actually the most important for us to recognize in this
setting. So pretending we never saw them would drastically reduce the effectiveness of our performance.

### Results

#### k ​ = 0.001 k ​ = 0.01 k ​ = 0.1 k ​ = 1

#### q ​ = 1/8 0.8823 0.8814 0.8709 0.

#### q ​ = 1/6 0.8606 0.8542 0.8540 0.

#### q ​ = 1/4 0.7889 0.7962 0.7967 0.

#### q ​ = 1/2 0.6898 0.6934 0.7009 0.

```
Figure 4: HMM Grid Search Accuracies - k vs. q
```
#### Although it seems like decreasing ​ q ​ any more would increase the training accuracy, the model is actually

overfitting to the training data. The ​ q ​ value represents the penalty applied to unknown words. Even

though we split randomly into training and validation sets, they both came from the same underlying

distribution of sentences. Therefore we can be confident in the model generalizing well with ​ k ​ = 0.01 and q ​ = 1/6.

On the Kaggle test set, our HMM implementation achieved ​ **70.7% accuracy** ​​ which is an ​ **improvement of
3.7% from our baseline system** ​​. Although we expected a larger margin, Professor Cardie mentioned in
class that the technique we used for our baseline is actually widely known to perform well in a wide range
of named entity recognition tasks. Similarly, an example given in the textbook, WSJ training corpus
achieves 92.4% accuracy from the most-frequent-tag baseline while the state of the art in part-of-tagging
on this dataset is around 97%, an accuracy achieved by HMM. Therefore, from 67% to 70.7% is a
reasonable improvement from the most-frequent-tag baseline to HMM.
One error that we noticed our HMM making was falsely predicting the MISC tag when the gold standard
was actually O. This could be due to the sparsity of MISC tags in the training corpus and the wide
distribution of MISC entities in language.

## MEMM Implementation

We extracted features from a window of words that appear before and after the current word we are trying
to predict. These features consist of one-hot encodings of the part of speech tags of the current word and
the previous/next ones in the window, along with the predicted tags of the previous words. It is important
for these features to be one-hot because any two tags should be the same pairwise distance apart.
Suppose we have 9 possible tags that could be predicted, 54​ ​unique parts of speech, and a window size of ​ w ​. 
Then the size of each input vector is ​ 54*(2w+1)+9w ​. 
We were able to vary this window size ​ w ​ in order to optimize performance on the validation set (which will be discussed in the experiments section).


We infer that the window size for features is a convex optimization problem as when we start getting too
far away from the target word, some of the features become irrelevant and outweigh the others.
We used the ​ sklearn ​ library as a resource for our maximum entropy model. In particular, we utilized the
LogisticRegression ​ model from the ​ linear_model ​ package. We created our initial implementation using the

default metaparameters but will discuss our hyperparameter optimization methods in the Experiments
section. The function inputs the feature representation described in the last paragraph and outputs 9
values, corresponding to a probability for each of the 9 possible tags.
We only had to change one line of our Viterbi function from the HMM to adapt it to the MEMM setting.
This one change was replacing the transition/emission probabilities in each iteration to the inference made by the ​ LogisticRegression ​ model.

### Experiments

#### Expectation/Hypothesis: ​ We expect the MEMM implementation to outperform the baseline system as

well. It also has a good chance of beating the HMM since we can now use a wider window and
incorporate the part of speech features too.
We used the same 70/30 train/validation split for our MEMM tests as well in order to be able to compare
with our HMM implementation.
One of the most important feature variations to optimize in this setting is the window size. Here we
change up the feature sets by including in increasing number of one-hot tags and past predictions for the
feature vectors. According to the Markov assumption, the current state only depends on a short sequence
of previous events. So it is important for us to figure out the optimal size of this sequence (results in
Figure 5).
Other variations in the model could be made by tweaking the metaparameters of the logistic regression
classifier. The one that was most interesting to us was the regularization technique. In particular, the
effect of using ​ L1 ​ vs. ​ L2 ​ regularization with varying ​ C ​ constants (results in Figure 6).

### Results

w = 2 w = 3
0.3461 0.
_Figure 5: Accuracies for window sizes on default LR parameters_
As we can see, training on a window size of 2 does not give us enough information to make meaningful
predictions. Increasing the window size allows us to incorporate more relevant feature information to help
us make predictions. We could have tried larger window sizes but due to computational restrictions, the
training time started taking way too long.
_L1 L_


##### 0.7467 0.

_Figure 6: Grid search for regularization type on optimal window size_
L1 ended up giving us the highest validation accuracy. This type of regularization is less sensitive to
outliers and more efficient on sparse feature vectors which is what we trained it on.
We noticed an error in several places that the MEMM made where an impossible combination of tags was
produced. For example we encountered a tagging of B-PER followed by an I-ORG, which is clearly
impossible. This error is most likely due to the discriminative nature of the algorithm; all words in the
window are incorporated in every feature uniformly rather than focusing on generating a reasonable
sequence.

### Comparing HMM and MEMM

We inferred that the error we encountered in the HMM was due to a low representation of MISC labels in
the training set, so in this case the error would also persist in the MEMM setting. However, the
sequencing error we found in the MEMM is solved in the HMM setting. We would almost never predict
I-ORG after seeing B-PER because the transition probability P(I-ORG | B-PER) ≈ 0, which would send the
product of the emission probability to 0. In this setting, we noticed that HMM’s performed better than MEMM’s. 
Our HMM got an accuracy of 85.4% while the MEMM got an accuracy of 74.67%. The HMM model only considers combinations of
consecutive taggings that actually make sense in a real setting. This essentially narrows down the
possible predictions by a half which seems more promising than the MEMM model that previously
showed sporadic behavior. Also, our HMM clearly overfit to the training set. In our grid search for hyperparameters, we noticed a
validation accuracy of up to 88% even though our Kaggle score was only 70.7%. This implies that the
hyper-parameters we chose fit very tightly to the distribution of the training set but does not generalize well in the wild. One of these hyper-parameters was a low ​ q ​ score which essentially kills the probabilities of unknown words. On the other hand, the generalizing MEMM hyper-parameter was the window size, for which we chose a large value. Therefore, our MEMM generalizes better so we still believe that it is the
superior model.



