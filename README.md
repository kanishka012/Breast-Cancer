Problem Statement:
Build and deploy an interactive web application that demonstrates 6 different classification algorithms on a single dataset. The application should allow users to:
Upload test data for predictions
Compare model performance across multiple metrics
Visualize results through confusion matrices
Make real-time predictions using trained models

Dataset Description: Attribute	Description
- Source:	Sklearn Built-in Dataset
- Instances:	569 samples
- Features:	30 numerical features
- Target:	0 → Malignant , 1 → Benign  
- Class Distribution:	(37.3%) Malignant cases, (62.7%) Benign cases


Feature list:
- Mean radius
- Mean texture
- Mean perimeter
- Mean area
- Mean smoothness
- And several other statistical measurements


Models & Performance
-Six classification models were implemented and evaluated: Logistic regression, decision tree, k-nearest neighbours, naive bayes, random forest, xgboost
 -------------------------------------------------------------------------
  Model        Accuracy     AUC      Precision   Recall   F1 Score  MCC                                                                      
  ------------ ------------ -------- ----------- -------- --------- ------ 
  XGBoost      0.9561       0.9901   0.9467      0.9861   0.966    0.9058   

  Random       0.9561       0.9929   0.9589      0.9722   0.9655   0.9054   
  Forest                                                                     

  KNN          0.9561       0.9788   0.9589      0.9722   0.9655   0.9054   

  Naive Bayes  0.8594       0.8517   0.4844      0.7209   0.5794   0.5131   

  Decision     0.8947       0.8968   0.9412      0.8889   0.9143   0.7803 
  Tree
  
  Logistic
  Regression   0.9825      0.9954    0.9861     0.9861   0.9861    0.9623
  
  -------------------------------------------------------------------------

Key Observations:
Logistic Regression	- Works very well due to clear linear separation in dataset. It provides stable and interpret-able results.
Decision Tree	      - Decision Tree captures non-linear patterns but may over-fit the training data. It is prone to over fitting, especially when not properly pruned.
KNN	                - Performs well after scaling but is computationally expensive.
Naive Bayes	        - Fast model but assumes feature independence.
Random Forest	      - High performance due to ensemble learning and reduced over fitting.
XGBoost             - Generally achieves the best performance due to boosting technique.
