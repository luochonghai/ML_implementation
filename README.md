Implementation_for_machine_learning_algorithms
====
1.Description
----
      Implementation for watermelon book's coding homework.
2.Log
----
180401: logistic_regression(dataset:ZhihuaZhou "Machine Learning" P89)
<br>use sklearn to test its performance first,then use numpy to implement the alogrithm via normal/stochastic grandient descent methods.
<br><br>180401: cross_validation vs leave_one_out(dataset:Iris;Tic-tac-toe)
<br>use sklearn's logistic regression to compare 10-fold cross validation&leave-one-out methods.
<br><br>180401: linear_discriminant_analysis(dataset:ZhihuaZhou "Machine Learning" P89)
<br>use sklearn to test its performance first, then use numpy to implement the algorithm via normal/outlier(X[15])-deleted LDA.
<br><br>180406:ID3_decision_tree(dataset:ZhihuaZhou "Machine Learning" P84:note that the data of 0.0267 should be modified to 0.267)
<br>use pandas to implement the algorithm, show data_distribution_pic by seaborn and illustrate the decision tree structure via graphviz(both in Chinese).
<br><br>180407:CART_decision_tree(dataset:ZhihuaZhou "Machine Learning" P80)
<br>use pandas to implement the algorithm, show the strucutre of "full" tree,the prepruned version and the postpruned version via graphviz.
<br><br>180421:standard back propagation algorithm
<br>implement BP algorithm and train an xor NN.
<br><br>180423:use SVM to classify the dataset of breast cancer,and compare its result with BP algorithm.
<br>implement SVM via sklearn, and implement BP algorithm via PyBrain.
<br><br>181204:compare the difference between func of normal distribution & sigmoid func, to find out why sigmoid func is widely used at the very beginning.
<br>Figure1: sigmoid_vs_normal_distir(func)
<img src="https://github.com/luochonghai/ML_implementation/blob/master/BP/sigmoid_vs_normal_distir(func).png"  alt="sigmoid_vs_normal_distir(func)"/>
<br>Figure2: sigmoid_vs_normal_distir(derivative)
<img src="https://github.com/luochonghai/ML_implementation/blob/master/BP/sigmoid_vs_normal_distri(derivative).png"  alt="sigmoid_vs_normal_distir(derivative)"/>
