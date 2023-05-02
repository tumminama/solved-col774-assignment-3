Download Link: https://assignmentchef.com/product/solved-col774-assignment-3
<br>
<ol>

 <li><strong>Decision Trees (and Random Forests):</strong></li>

</ol>

Machine learning has been deployed in various domains of computer science as well as other engineering fields. In this problem you will work on detecting files infected with virus on your system. You will work with <a href="https://archive.ics.uci.edu/ml/datasets/Dynamic+Features+of+VirusShare+Executables">VirusShare</a> dataset available for download from the UCI repository. Read about the dataset in detail from the link given above. You have been provided with a pre-defined set of test, train and validation of dataset to work with (available for download from the course website). In this dataset, for any given example, a large number attribute values are missing (they can be thought of having as a ’default’ value,<em>i.e. </em>0). Correspondingly, you have also been provided sparse files and you can use <a href="https://github.com/kunaldahiya/pyxclib"><strong>pyxclib</strong></a> for reading and

writing the sparse files. You have to implement the decision tree algorithm for predicting the virus infected files based on a variety of features. You will also experiment with Random Forests in the last part of this problem.

<ul>

 <li><strong> Decision Tree Construction </strong>Construct a decision tree using the given data to predict which files are infected. You should use mutual information as the criteria for selecting the attribute to split on. At each node, you should select the attribute which results in maximum decrease in the entropy of the class variable (i.e. has the highest mutual information with respect to the class variable). This problem has all its attributes as integer (continuous) valued. For handling continuous attributes, you should use the following procedure. At any given internal node of the tree, a numerical attribute is considered for a two way split by calculating the median attribute value from the data instances coming to that node, and then computing the information gain if the data was split based on whether the numerical value of the attribute is greater than the median or not. For example, if you have have 10 instances coming to a node, with values of an attribute being (0,0,0,1,1,2,2,2,3,4) in the 10 instances, then we will split on value 1 of the attribute (median). Note that in this setting, a numerical attribute can be considered for splitting multiple number of times. At any step, choose the attribute which results in highest mutual information by splitting on its median value as described above. Note that a large number of attribute values are missing for any given instance, and for this problem, it is safe to treat them as having a default value of ’0’ <a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>. Plot the train, validation and test set accuracies against the number of nodes in the tree as you grow the tree. On X-axis you should plot the number of nodes in the tree and Y-axis should represent the accuracy. Comment on your observations.</li>

 <li><strong>Decision Tree Post Pruning </strong>One of the ways to reduce overfitting in decision trees is to grow the tree fully and then use post-pruning based on a validation set. In post-pruning, we greedily prune the nodes of the tree (and sub-tree below them) by iteratively picking a node to prune so that resultant tree gives maximum increase in accuracy on the validation set. In other words, among all the nodes in the tree, we prune the node such that pruning it(and sub-tree below it) results in maximum increase in accuracy over the validation set. This is repeated until any further pruning leads to decrease in accuracy over the validation set. Read the <a href="https://poorvi.cse.iitd.ac.in/~parags/teaching/2016/col774/notes/dtree_pruning_mitchell.pdf">following notes</a> on pruning decision</li>

</ul>

trees to avoid overfitting (also available from the course website). Post prune the tree obtained in step (a) above using the validation set. Again plot the training, validation and test set accuracies against the number of nodes in the tree as you successively prune the tree. Comment on your findings.

<ul>

 <li><strong> Random Forests: </strong>As discussed in class, Random Forests are extensions are decision trees, where we grow multiple decision trees in parallel on bootstrapped samples constructed from the original training data. A number of libraries are available for learning Random Forests over a given training data. In this particular question you will use the scikit-learn library of Python to grow a Random Forest. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">Click here</a> to read the documentation and the details of various parameter options. Try growing different forests by playing around with various parameter values. Especially, you should experiment with the following parameter values (in the given range): (a) <em>n estimators </em>(50 to 450 in range of 100). (b) <em>max features </em>(0.1 to 1.0 in range of 0.2) (c) <em>min samples split </em>(2 to 10 in range of 2). You are free to try out non-default settings of other parameters too. Use the out-of-bag accuracy (as explained in the class) to tune to the optimal values for these parameters. You should perform a <a href="https://scikit-learn.org/stable/modules/grid_search.html">grid search</a> over the space of parameters (read the description at the link provided for performing</li>

</ul>

grid search). Report training, out-of-bag, validation and test set accuracies for the optimal set of parameters obtained. How do your numbers, i.e., train, validation and test set accuracies compare with those you obtained in part (b) above (obtained after pruning)?

<ul>

 <li><strong> Random Forests – Parameter Sensitivity Analysis: </strong>Once you obtain the optimal set of parameters for Random Forests (part (c) above), vary one of the parameters (in a range) while fixing others to their optimum. Plot the validation and test Repeat this for each of the parameters considered above. What do you observe? How sensitive is the model to the value of each parameter? Comment.</li>

 <li><strong>Extra Fun: No Credits!</strong>: Read about the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html">XG-boost</a> algorithm which is an extension of decision trees to Gradient Boosted Trees. You can read about gradient boosted trees <a href="https://towardsdatascience.com/machine-learning-part-18-boosting-algorithms-gradient-boosting-in-python-ef5ae6965be4">here (link 1)</a> and <a href="https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning">here (link 2)</a><a href="https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning">.</a></li>

</ul>

Try out using XG-boost on the above dataset. Try out different parameter settings, and find the one which does best on the validation set. Report the corresponding test accuracies. How do these compare with those reported for Random Forests?

<ol start="2">

 <li><strong>Neural Networks :</strong></li>

</ol>

In this problem, you will work with the English Alphabets Dataset available for download from the course website. This is an image dataset consisting of greyscale pixel values of 28×28 size images of 26 Alphabets. Therefore this is a multiclass dataset containing 26 classes. The Dataset contains a train and a test set consisting of 13000 and 6500 examples, respectively. The last entry in each row denotes the class label. Note that the dataset is balanced; i.e. the number of examples is the same for all classes in train and test set. In this problem, we will use what is referred to a one-hot encoding of the output labels. Given a total of <em>r </em>classes, the output is represented an <em>r </em>sized binary vector (i.e., <em>y </em>∈R<em><sup>r</sup></em><sup>×1</sup>), such that each component represents a Boolean variable, i.e., <em>y<sub>l </sub></em>∈{0<em>,</em>1}, ∀<em>l,</em>1 ≤ <em>l </em>≤ <em>r</em>). In other words, each <em>y </em>vector will have exactly one entry as being 1 which corresponds to the actual class label and all others will be 0. This is one of the standard ways to represent discrete data in vector form <a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>. Corresponding to each output label <em>y<sub>l</sub></em>, our network will produce (independently) an output label <em>o<sub>l </sub></em>where <em>o<sub>l </sub></em>∈ [0<em>,</em>1].

<ul>

 <li><strong> </strong>Write a program to implement a generic neural network architecture to learn a model for multi-class classification using one-hot encoding as described above. You will implement the backpropagation algorithm (from first principles) to train your network. You should use mini-batch Stochastic Gradient Descent (mini-batch SGD) algorithm to train your network. Use the Mean Squared Error(MSE) over each mini-batch as your loss function. Given a total of <em>m </em>examples, and <em>M </em>samples in each batch, the loss corresponding to batch # <em>b </em>can be described as:</li>

</ul>

(1)

Here each <em>y</em><sup>(<em>i</em>) </sup>is represented using one-hot encoding as described above. You will use the sigmoid as activation function for the units in <strong>output </strong>layer as well as in the hidden layer (we will experiment with other activation units in one of the parts below). Your implementation(including back-propagation) MUST be from first principles and not using any pre-existing library in Python for the same. It should be generic enough to create an architecture based on the following input parameters:

<ul>

 <li>Mini-Batch Size (<em>M</em>)</li>

 <li>Number of features/attributes (<em>n</em>)</li>

 <li>Hidden layer architecture: List of numbers denoting the number of perceptrons in the corresponding hidden layer. Eg. a list [100 50] specifies two hidden layers; first one with 100 units and second one with 50 units.</li>

 <li>Number of target classes (<em>r</em>)</li>

</ul>

Assume a fully connected architecture i.e., each unit in a hidden layer is connected to every unit in the next layer.

<ul>

 <li><strong> </strong>Use the above implementation to experiment with a neural network having a <strong>single </strong>hidden layer. Vary the number of hidden layer units from the set {1, 5, 10, 50, 100}. Set the learning rate to 0.1. Use a mini-batch size of 100 examples. This will remain constant for the remaining experiments in the parts below. Choose a suitable stopping criterion and report it. Report and plot the accuracy on the training and the test sets, time taken to train the network. Plot the metric on the Y axis against the number of hidden layer units on the X axis. What do you observe? How do the above metrics change with the number of hidden layer units? NOTE: For accuracy computation, the inferred class label is simply the label having the highest probability as output by the network.</li>

 <li><strong> </strong>Use an adaptive learning rate inversely proportional to number of iterations i.e. <em>η<sub>t </sub></em>= <em><sup>η</sup><sub>t</sub></em><u><sup>0 </sup></u>where <em>η</em><sub>0 </sub>= 0<em>.</em>5 is the seed value and <em>t </em>is the current iteration number. Note that <em>t </em>gets incremented at very step of SGD update (i.e., once for every mini-batch). See if you need to change your stopping criteria. Report your stopping criterion. As before, plot the train/test set accuracy, as well as training time, for each of the number of hidden layers as used in <strong>?? </strong>above using this new adaptive learning rate. How do your results compare with those obtained in the part above? Does the adaptive learning rate make training any faster? Comment on your observations.</li>

 <li>Several activation units other than sigmoid have been proposed in the literature such as tanh, and RelU to introduce non linearity into the network. ReLU is defined using the function: g(z) = max(0, z). In this part, we will replace the sigmoid activation units by the ReLU for all the units</li>

</ul>

in the hidden layers of the network (the activation for units in the output layer will still be sigmoid to make sure the output is in the range (0,1)). You can read about relative advantage of using the ReLU over several other activation units <a href="https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/">on this blog</a><a href="https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/">.</a>

Change your code to work with the ReLU activation unit. Note that there is a small issue with ReLU that it non-differentiable at <em>z </em>= 0. This can be resolved by making the use of sub-gradients intuitively, sub-gradient allows us to make use of any value between the left and right derivative at the point of non-differentiability to perform gradient descent see this <a href="https://en.wikipedia.org/wiki/Subderivative">(Wikipedia page</a> for more details).

Implement a network with 2 hidden layers with 100 units each. Experiment with both ReLU and sigmoid activation units as described above. Use the adaptive learning rate as described in part2c above. Report your training and test set accuracies in each case. Also, make a relative comparison of test set accuracies obtained using the two kinds of units. What do you observe? Which ones performs better? Also, how do these results compare with results in part 2b using a single hidden layer with sigmoid. Comment on your observations.

<ul>

 <li><strong>(5 points) </strong>Use <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html">MLPClassifier from scikit-learn library</a> to implement a neural network with the same</li>

</ul>

architecture as in Part 2d above, and same kind of activation functions (ReLU for hidden layers, sigmoid for output). Use Stochastic Gradient Descent as the solver. Note that MLPClassifier only allows for Cross Entropy Loss over the final network output. Use the binary cross entropy loss for the multi-class classification problem. Here, we have a binary prediction for each possible output label, and apply a cross entropy loss over each such prediction. Note that in this formulation, there are <em>r </em>such outputs of the network, one for each label, and cross entropy loss is applied for each of them (and then added to compute the final loss). Read about <a href="https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a">the binary cross entropy loss</a> here<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a> (you should

refer to the two class entropy loss formulation only, since we have transformed our multi-class problem into <em>r </em>two class (problems), one for each label). Compare the performance with the results of Part 2d. How does training using existing library (and modified loss function) compare with your results in Part 2d above.

<ul>

 <li><strong>(Extra Fun, No Credits!) </strong>Modify your loss function in Part 2d above (your own implementation) to be binary cross entropy loss as in Part 2e above. How do your results change? What is the impact of using a different loss function in your implementation? Also compare with results obtained in Part 2e. After this change, the only difference remaining is between your own implementation and the library implementation of the learning algorithm (since the network architecture as well as the loss functions are now the same). Does prediction accuracy (train/test) of your implementation compare with the library implementation? Why or why not? What could the possible reasons for differing accuracy numbers?</li>

</ul>

<a href="#_ftnref1" name="_ftn1">[1]</a> think about why this makes sense. Are there any other ways that you can deal with these missing attribute values? Feel free to try alternate methods

<a href="#_ftnref2" name="_ftn2">[2]</a> Feel free to read about the one-hot encoding of discrete variables online

<a href="#_ftnref3" name="_ftn3">[3]</a> you can also refer to the lecture notes from April 10 – section on GANs to know more about cross entropy loss