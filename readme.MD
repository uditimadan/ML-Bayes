### Assignment 3: Machine Learning with Naive Bayes
#### Due Mon 3/18  at 11:59pm. 

Note: For the programming  portions of the assignment, please provide a file called submission.py that demos your code.
For the written portions, please prepare a document called assignment3.pdf and put your answers in there.
And don't forget to put your name on your assignment!

I've provided some starter code for you. You are welcome to adapt it as you like. I've used pandas, and encourage you to do the same.

**On time management!** This assignment has many small components; the key to success will be starting early.
I've added suggested milestones _in italics_. 

_Feb 28_
1. (10 points) In the Feb 23 asynchronous lecture, you were asked to complete four questions about probability. Please include the answers to those questions in the PDF you submit.

**Intro to sklearn.**

In this question, you'll get a basic understanding of how to use scikit-learn. 

I've provided a starter file that loads in the Iris dataset and uses a Categorical Naive Bayes classifier to classify flowers. 

_March 1_
2. (5 points) Change this to use the wine dataset.

_March 2_
3. (5 points) 5-fold cross-validation. As we discussed in class, a challenge in evaluating the performance of a learning algorithm is constructing 
training and test sets that are separate from each other, but also large enough to give useful results. The standard way to solve this is through
*cross-validation*. 

For this part, you should add in the [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score)
method from sklearn to perform cross-validation.

_March 3_

4. (5 points) Evaluating performance. We also have seen that choosing the right way to evaluate the effectiveness of an ML algorithm can be 
complex. By default, cross_val_score uses _accuracy_ as a measure. Change this to use F1 through the use of the 'scoring' keyword argument. 

** Implementing Naive Bayes ** 

Now that you have a sense for how to use the Naive Bayes classifier in sklearn, we're going to implement it ourselves. 
I've provided a template for you in nb.py. 

We will use the same breast cancer dataset from Assignment 1. Our goal will be to build a classifier that can, for unseen data, predict whether a patient will have a recurrence.

_March 5_

5. (15 points) To begin, we need to write fit. This should take as input a list of examples and a list of classifications (just like sklearn)
and generate a model. In this case, our model is a set of conditional probabilities, stored as counts. 

That is, you'll store each P(feature-value | classification) in one of the two dictionaries. The easiest way to do this is by getting the value_counts.
Please see the template code for more details.

_March 5_

6. (5 points) Next, write a unit test for fit. (I've provided a stub). How would you know whether your method is correct?

_March 7_

7. (15 points) Next we'll implement predict. Predict should take a list of unlabeled instances (named examples in the input) and, for each
example, use Naive Bayes to predict that instance's classification. 
Recall that what we want to know whether P(recurrence | feature1,...,featuren) > p(no-recurrence | feature1,...,featuren)

To compute this, we use the Naive Bayes assumption and use P(feature1 | recurrence) * P(feature2 | recurrence) * ... * P(featuren | recurrence) * P(recurrence)

We actually need to compute log-likelihood in order to handle underflow, so fit should compute:
log(P(feature1 | recurrence)) + log(P(feature2 | recurrence)) + ... + log(P(recurrence))

(and the same for no-recurrence)

_March 7_

8. (5 points) Next, write a unit test for predict. It should fit a model, test a known example, and assert whether the right value is returned.

_March 8_

9. (10 points) Now we want to implement score. Score should take as input two lists or Series representing predicted and actual, and compute F1 for them.

_March 8_

10. (5 points) Again, add a unit test confirming that this works. Try it with lists that are identical, completely different, and mostly right.

_March 10_

11. (15 points) Now let's implement five-fold cross-validation. 
 
- Split the data into five equal "bins". (If the number is not divisible by 5, that's fine. Some bins can have one extra item.)
 Note that you should not actually copy the data into new structures - just use indices to keep track of which data is for training and which is for testing.

You'll do five iterations - for each iteration, 4 of the bins are training, and 1 is the test bin. In each iteration, create a new classifier and fit it to the training data.
Then test that classifier on the test data and compute F1. Once you're done, return the five F1 scores.

12. (5 points) Lastly, compare your classifier to ZeroR on the breast cancer data, using five-fold cross-validation for each, and include a table in your PDF showing the results.

**(grad students only):**

Please read [this article](https://12ft.io/proxy?q=https%3A%2F%2Fwww.theatlantic.com%2Fmagazine%2Farchive%2F2013%2F11%2Fthe-man-who-would-teach-machines-to-think%2F309529%2F) about Douglas Hofstadter, which also serves as a nice summary of the history of AI and the debates over the value of developing machines that think like humans.

(As an aside: If you have not read Hofstadter's book [Godel, Escher, Bach](https://en.wikipedia.org/wiki/G%C3%B6del,_Escher,_Bach), I strongly recommend it.)

Prepare a summary or critique of this article that addresses the following questions:

Hofstadter is particularly interested in understanding the way humans think. What sorts of reasoning mechanisms does he study?
The article includes a quote from our text: “The quest for ‘artificial flight’ succeeded when the Wright brothers and others stopped imitating birds and started … learning about aerodynamics,” What does this mean? Why is it relevant to AI?
What was Candide? Why did it change the way we thought about machine translation?
The article also contains a quote from the last chapter of AIMA: perhaps AI has become too much like the man who tries to get to the moon by climbing a tree: “One can report steady progress, all the way to the top of the tree.” 
What does this mean? How does it relate to Candide and the ways in which big data and machine learning have changed AI?


