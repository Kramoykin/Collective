# Data Analysis – student alcohol consumption dataset exploration
<p align="justify">
  
This is the repo of Tomsk Politechnic University students course work on the field of data analysis. The [chosen dataset](https://www.kaggle.com/uciml/student-alcohol-consumption) contains a survey of students math and portuguese language courses in secondary school: lot of interesting social, gender and study information about students. 

The goal of this work is to go through such data analysis steps as **data preparation**, **descriptive** and **explanatory** statistics and finally – **predictive analysis** with using several predive models. 
</p>

# Introduction
<p align="justify">
  
As part of the study of the subject "Introduction to Big Data Analysis", the influence of various social, psychological, and property factors on students' academic performance was investigated. The study was conducted based on data [https://www.kaggle.com/uciml/student-alcohol-consumption] using statistical methods and machine learning algorithms.
The work aimed to acquire knowledge and skills of working with data using statistical analysis and the Python programming language with NumPy, Pandas, sklearn data processing libraries
The study **aims** to quantify the relationship between the influence of various factors on student academic performance.

</p>

* To describe the data under study statistically and graphically to obtain detailed information about the object of study
* To carry out the stage of data preparation for further analysis
* Identify the relationships between the parameters and select the most significant
* Perform predictive analysis based on the prepared data using Logistic Regression and Random Forest algorithms
* Evaluate the results of predictive analysis using accuracy metrics and ROC-AUC estimates.

## Data preparation and descriptive analysis
<p align="justify">
  
The dataset was explored for future investigation by using __Pandas__ functional. It were founded that existing dataset is pretty clear and does not contain NaN or Null values. For the plotting process optimization attributes were divided in three groups:
</p>

* categorical; 
* discrete;
* continuous.

<p align="justify">
  
There are some differences in representation of each group attributes distibutions. Due to the different amount of students in Math course and Portuguese courses histograms were normalized on the amount of students to be compared easier. As example you can see below normalized __seaborn__ countplots of daily alcohol consumption (Dalc) and weekend alcohol consumption (Walc).
</p>

<div align="center">
  
Daily consumption distribution             |  Weekend consumption distribution 
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/63719570/140008238-e6d5d509-d0bf-46c5-8211-48f618ef6657.png)  |  ![image](https://user-images.githubusercontent.com/63719570/140008191-730afa96-e9aa-4f58-9d3f-c3ffcda78fb2.png)
</center>

</div>
  
<p align="justify">
  
For continuous variables exists **seaborn** distplots like following normalized distribution of the final grade:
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/63719570/140008932-b497022b-57c2-45b3-b3af-7e1f544afac4.png" />
</p>

Also on the descriptive stage was implemented the ability to construct correlation matrix  for all parameters:

<p align="center">
  <img src="https://user-images.githubusercontent.com/63719570/140009996-6347c8a6-42df-4657-8a70-5d3cb3dba807.png" />
</p>

## Predictive analysis

### Logistic Regression model

<p align="justify">
For the following analysis both Math and Portuguese datasets were merged with removing duplicated rows, appropriate for students attending both courses. For proper using of the logistic regression the target variable (final grade - G3) should have been transformed. On the following picture
<p align="center">
  <img src="https://user-images.githubusercontent.com/63719570/140011262-2362d045-911c-4170-a075-53e63b346d24.png" />
</p>

<p align="justify">
you can clearly notice the distinct border between G3 = 9 and G3 = 10. It was decided to call all students with G3 less or equal – 9 'bad', and with G3 greater than 9 – 'good'. 

Categorical variables were transformed to the binary format and it was explored the feature importance of all attributes by using **sklearn.DecisionTreeClassifier**. Features with zero importance were removed from dataset. The dataset was randomly splitted into train, test and validation samples.  Train and test samples were used for Hyperparameter Tuning -- iterative applying the **sklearn.LogisticRegression** model with different values of C-parameter to get the best one value by calculating accuracy of the model and logarithmic loss. 

Finally the **sklearn.LogisticRegression** model with the best value of C-parameter was used on the validation sample. Calculated accuracy is 0.92 and logarithmic loss is 0.19. Results of the prediction are shown in confusion matrix:
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/63719570/144796048-d251ba7e-d273-4049-bdad-85425333a0ed.png" />
</p>

<p align="justify">
Efficiency of Logistic Regression application was also estimated by using ROC/AUC metric:   
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/63719570/144796415-d0d7920e-64bd-4abc-a47d-50888f042b5a.png" />
</p>

<p align="justify">
During the roc/auc exploration it was calculated list of treshold values for Logistic Regression Classification. It was implemented the treshold adjustment by the accuracy value corresponding to certain trehold value. The confusion matrix after adjustment is represented below:
<p align="center">
  <img src="https://user-images.githubusercontent.com/63719570/144796954-76a9ccff-444d-44e7-9c5d-f012f0e3231a.png" />
</p>
but accuracy value is still the same -- 0.92.
<p align="justify">  
With the aim of verifying the model there was used __sklearn.DummyClassifier__ which results of accuracy is 0.71 and of logarithmic loss is 9.87 are much worse.

Also it was used some **SHAP** functional for describing the resulting feature importance of used attributes:
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/63719570/140012756-02a56362-e21a-402f-9340-d028edd99009.png" />
</p>

### Random Forest model

It was also created Random Forrest classification model for student's academic performance. All the data preparation, features engineering and creating train/test/valid  samplings operations were the same. The Random Forest model with the parameters n_estimators == 100 and criterion == 'entropy' was fitted on the train data. Testing accuracy value on validation was 0.88 with the following confusion matrix:
<p align="center">
  <img src="https://user-images.githubusercontent.com/63719570/144799510-c4607d16-8761-4505-a72f-ffe4d66252e1.png" />
</p>

After that it was implemented max_features and max_depth hyperparameters tuning algorithm. Efficiency of the tuning was explored by the confusion matricies and accuracy values. It was founded that the best results were given by default hyperparameter values.  

The roc/auc analysis gave the following results:
<p align="center">
  <img src="https://user-images.githubusercontent.com/63719570/144799699-befa5a18-313d-4f0b-b6cc-bc72e5900e86.png" />
</p>

Treshold value was adjusted by the same algorithm as it was for Logistic Regression. The accuracy value on the validation set is 0.91 and confusion matrix:

<p align="center">
  <img src="https://user-images.githubusercontent.com/63719570/144817002-a420c286-6f2d-4f41-b670-0748fc304e5f.png" />
</p>

# Conclusion
<p align="justify">
  
As a result of the study, a statistical description of the data [https://www.kaggle.com/uciml/student-alcohol-consumption] and a set of graphs and histograms reflecting the features of the distribution and interrelation of parameters were obtained.
After preparing the data and combining datasets through the calculation of entropy on binary trees, the parameters that most affect students' academic performance were selected. Using the selected parameters as predictors, a predictive analysis was carried out using algorithms of Logistic Regression and Random Forest.
The evaluation of the results of the models was carried out based on accuracy criteria and ROC-AUC score. The accuracy values were 0.92 for Logistic Regression and 0.91 for the random forest. The AUC score showed values of 0.979 for Logistic Regression and 0.957 for Random Forest.
Based on the estimates obtained, it can be concluded that the effectiveness of the models for the selected data differs slightly. The sufficiently high accuracy of the predictions obtained shows that the selected parameters largely determine the academic performance of students.

</p>

## Useful Links:
**Dataset** : 

https://www.kaggle.com/uciml/student-alcohol-consumption

**Data handling**

https://pandas.pydata.org/

https://numpy.org/

**Plotting** :

https://matplotlib.org/

https://seaborn.pydata.org/

**Machine learning** :

https://scikit-learn.org/stable/

**Explaining model results** :

https://shap.readthedocs.io/en/latest/index.html
