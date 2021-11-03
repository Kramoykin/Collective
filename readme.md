# Data Analysis – student alcohol consumption dataset exploration
<p align="justify">
This is the repo of Tomsk Politechnic University students course work on the field of data analysis. The [chosen dataset](https://www.kaggle.com/uciml/student-alcohol-consumption) contains a survey of students math and portuguese language courses in secondary school: lot of interesting social, gender and study information about students. 

The goal of this work is to go through such data analysis steps as **data preparation**, **descriptive** and **explanatory** statistics and finally -- **predictive analysis** with using several predive models. 
</p>

## Data preparation and descriptive analysis
<p align="justify">
The dataset was explored for future investigation by using **Pandas** functional. It were founded that existing dataset is pretty clear and does not contain NaN or Null values. For the plotting process optimization attributes were divided in three groups:
</p>

* categorical; 
* discrete;
* continuous.

<p align="justify">
There are some differences in representation of each group attributes distibutions. Due to the different amount of students in Math course and Portuguese courses histograms were normalized on the amount of students to be compared easier. As example you can see below normalized **_seaborn_** countplots of daily alcohol consumption (Dalc) and weekend alcohol consumption (Walc).
</p>

<center> 
Daily consumption distribution             |  Weekend consumption distribution 
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/63719570/140008238-e6d5d509-d0bf-46c5-8211-48f618ef6657.png)  |  ![image](https://user-images.githubusercontent.com/63719570/140008191-730afa96-e9aa-4f58-9d3f-c3ffcda78fb2.png)
</center>
  
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

### Logistic regression model

<p align="justify">
For the following analysis both Math and Portuguese datasets were merged with removing duplicated rows, appropriate for students attending both courses. For proper using of the logistic regression the target variable (final grade - G3) should have been transformed. On the following picture
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/63719570/140011262-2362d045-911c-4170-a075-53e63b346d24.png" />
</p>

<p align="justify">
you can clearly notice the distinct border between G3 = 9 and G3 = 10. It was decided to call all students with G3 less or equal – 9 'bad', and with G3 greater than 9 – 'good'. 

Categorical variables were transformed to the binary format and it was explored the feature importance of all attributes by using **sklearn.DecisionTreeClassifier**. Features with zero importance were removed from dataset. The dataset was randomly splitted into train, test and validation samples.  Train and test samples were used for Hyperparameter Tuning -- iterative applying the **sklearn.LogisticRegression** model with different values of C-parameter to get the best one value by calculating accuracy of the model and logarithmic loss. 

Finally the **sklearn.LogisticRegression** model with the best value of C-parameter was used on the validation sample. Calculated accuracy is 0.92 and logarithmic loss is 0.19. Results of the prediction are shown in confusion matrix:
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/63719570/140012350-c621d27e-24c8-4cc5-9ad3-95034f489590.png" />
</p>

<p align="justify">
With the aim of verifying the model there was used **_sklearn.DummyClassifier_** which results of accuracy is 0.71 and of logarithmic loss is 9.87 are much worse.

Also it was used some **_SHAP_** functional for describing the resulting feature importance of used attributes:
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/63719570/140012756-02a56362-e21a-402f-9340-d028edd99009.png" />
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
