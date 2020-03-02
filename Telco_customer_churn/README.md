# Telecom customer churn

<p align="center">
<img src='https://www.superoffice.com/blog/wp-content/uploads/2015/05/customer-churn-750x400.jpg'></img>
</p>

### Introduction 

Customer segmentation is one of the most important metrices for growing companies to evaluate. While it's not the happiest measure, it's a number that give companies the hard truth about its customer retention strategy. 

**What is Customer Chrun?**<br>
Customer Churn is the percentage of customers that stopped using company's product/service during a certain time frame. The calculation for  **customer churn** is diving the number of customers company lost during the time period by the number of customers company had at the beginning of that time period.

For example, if company had 400 customers at the beginning of the year and end with 360, means company's churn rate is 10%, because compant lost 10% of the the customer from the base. Most companies try to decrease churn rate as 0%.


### Datasets, Features and Target value

**Source** : https://www.kaggle.com/blastchar/telco-customer-churn ( IBM Sample dataset)

Here, IBM provided customer data for Telco industry to predict behaviour of the customers. Main objective is that to analyze customer behavious and develop strategies for customer retention.

Dataset has information related,

**Demographic**: 
  - Gender - Male / Female
  - Age range - In terms of Partner, Dependent and Senior Citizen
  
**Services**: 
 - Phone service - If customer has Phone service, then services related to Phone like;
    - Multiline Phone service
 - Internet Service - If customer has Internet service, then services related to Internet like;
    - Online security
    - Online backup
    - Device protection
    - Tech support
    - Streaming TV
    - Streaming Movies
    
**Account type**:
- Tenure - How long customer is with the company?
- Contract type - What kind of contract they have with a company? Like 
    - Monthly bases
    - On going bases - If on going bases, then One month contract or Two year contract
- Paperless billing - Customer is paperless billion option or not?
- Payment method - What kind of payment method customer has?
    - Mailed check
    - Electronic check
    - Credit card (Automatic)
    - Bank transfer (Automatic)

**Usage**:
- Monthly charges
- Total charges

**Target**:
- Churn - Whether customer left the company or still with the company?

### Problem Statement

Main objective is to detect leaving customers and provide retention plans. In order to achieve this objective, divide whole things into three groups.

**1) Why customers leaving the company?**<br>
The reason behind the customer leaving the company could be,
1) High charges
2) Better offer from competitor
3) Poor customer service
4) Some unknown reason

**2) How to detect the churn customers?**<br>
- Monitoring usage
- Analysing complains
- Analysing competitors offers

**3) How to prevent customers to leave a company?**<br>
Once you detect the high risk customers, apply
- Retension plans
- Improve customer service

### Notebook

Link - https://github.com/ShivaliPatel/Data-science-projects/blob/master/Telco_customer_churn/Notebooks/Telecom%20Churn%20Prediction.ipynb

### **Content**
1) **Descriptive analysis and EDA (Exploratory data analysis)**
- Churn rate and correlation between features
- Profiler of Churn vs Existing customers
- Tenure and Monthly charges distribution
2) **Cluster analysis**
- Churn cluster detection
- Churn customer cluster analysis - by demographic, account type and service usage
3) **Churn customer prediction model**
- Predictive model process
  - Data preprocessing
  - Model selection
  - Hyperparameter tuning
  - Model evaluation 
 4) **Retention plan**


### 1) Descriptive analysis and EDA (Exploratory data analysis)

#### Churn rate and correlation between features
Analysis shows that Churn rate of the Telecom company is around 26%.

<p align="center">
<img src ='https://github.com/ShivaliPatel/Data-science-projects/blob/master/Telco_customer_churn/Images/Churn_rate.png' width =450 height=300 ></img>
</p>


From correlation matrix, features like Tenure, Monthly charges and Total charges are highly correlated with services like Multiple Phone Lines services and Internet services like Online Security, Online Backup, Device Protection, Tech Support, Streaming TV and Streaming Movies services.
<p align='center'>
<img src ='https://github.com/ShivaliPatel/Data-science-projects/blob/master/Telco_customer_churn/Images/Correlation_matrix.png' width = 700 height=600 ></img>
</p>

#### Profiler of Churn vs Existing customers

Churn customers are likely to<b>
- not have Partner and Dependents; Meaning likely to be single.
- have Internet service and specifically Fiber optics
- not have online security service, online backup service, device protection service, Tech support service
- have streaming TV and streaming Movies services
- be with monthly based plan
- have paperless billing service
- have electronic check payment method</b>

#### Tenure and Monthly charges distribution

From distribution, churn subscribers are 
- more likely to leave company who's tenure is less than a year 
- more likely to have more than $65 monthly charges 

<p align='center'>
<img src='https://github.com/ShivaliPatel/Data-science-projects/blob/master/Telco_customer_churn/Images/Tenure_distribution.png'
     width =550 height=400></img>
</p>

<p align="center">
<img src ='https://github.com/ShivaliPatel/Data-science-projects/blob/master/Telco_customer_churn/Images/MonthlyCharges_distribution.png' width =550 height=400 ></img>
</p>

### 2) Cluster analysis

Based on Monthly Charges and Tenure, there is three types of clusters.

- Low Tenure and Low Monthly Charges (Blue)
- Low Tenure and High Monthly Charges (Green)
- High Tenure and High Monthly Charges (Red)

Around 50% of churn customers belongs to Low tenure and High Monthly Charges.

<p align='center'>
<img src='https://github.com/ShivaliPatel/Data-science-projects/blob/master/Telco_customer_churn/Images/Churn_cluster_detection.png'
width=500 height=350></img>
</p>

Based on Demographic information,

Low Tenure and Low Monthly Charges customers 
- Male
- Dependents

Low Tenure and High Monthly Charges customers 
- Senior citizens
- Female

High Tenure and High Monthly Charges customers 
- Male
- Partner, Dependents and Senior Citizen

<p align='center'>
<img src='https://github.com/ShivaliPatel/Data-science-projects/blob/master/Telco_customer_churn/Images/Churn_cluster_by_demographic.png'
width=500 height=350></img>
</p>

Based on Account information,

Low Tenure and Low Monthly Charges customers 
- Month-to-month contract plan

Low Tenure and High Monthly Charges customers 
- Paperless billing
- Month-to-month contract plan

High Tenure and High Monthly Charges customers 
- Paperless billing
- One/Two year contract type

<p align='center'>
<img src='https://github.com/ShivaliPatel/Data-science-projects/blob/master/Telco_customer_churn/Images/Churn_cluster_by_account_info.png'
width=500 height=350></img>
</p>

Based on Usage information,

Low Tenure and Low Monthly Charges customers 
- Have DSL internet service

Low Tenure and High Monthly Charges customers 
- Have Streaming TV / Streaming Movies
- Fiber optic internet service

High Tenure and High Monthly Charges customers 
- Online services like Online Backup, Device Protection and Tech Support
- Fiber optic internet service
- Have Streaming TV / Streaming Movies

<p align='center'>
<img src='https://github.com/ShivaliPatel/Data-science-projects/blob/master/Telco_customer_churn/Images/Churn_cluster_by_usage_info.png'
width=500 height=350></img>
</p>

### 3) Churn customer prediction model

1) Data Preprocessing
- Splitting dataset into two groups – Training & Testing
- Class imbalance issue due to inequality in Existing and Churn customers distribution

2) Model Selection
- Comparing models like Logistic regression, Random forest using corss_val_score() method
- Measuring scores like Accuracy, Precision, Recall and F1 metrics

3) Parameter tuning
- Using GridSearchCV() method, find out best parameters for respected models

4) Model Evaluation
- Using Classification report & Log loss score calculate best model for our data

Here, main objective should be to detect all churn customers and retain them. In order to achieve that need to minimize Log loss value as well as improve Recall.
Both model has same log loss score as well as Recall. But random forest has higher precision, I would go with that.
Hence, we will use to predict churn customers based on Random Forest.

<p align='center'>
<img src='https://github.com/ShivaliPatel/Data-science-projects/blob/master/Telco_customer_churn/Images/Logistic_reg_evaluation_matrix.png'
width=300 height=350></img>
</p>


<p align='center'>
<img src='https://github.com/ShivaliPatel/Data-science-projects/blob/master/Telco_customer_churn/Images/Random_forest_evaluation_matrix.png'
width=300 height=350></img>
</p>


Random forest model suggested important features like
- Contract type
- Payment method
- Tenure
- Internet usage – Fiber optics
- Total & Monthly Charges
- Paperless billing
- Online security, Tech support

Most of them, we already analyzed during our EDA process.


<p align='center'>
<img src='https://github.com/ShivaliPatel/Data-science-projects/blob/master/Telco_customer_churn/Images/Random_forest_feature_importance.png'
width=700 height=450></img>
</p>

### 4) Retention plan

Since we generated a model based on Churn and Existing customers, which help to classify both of them. Now we can use same model on existing customers to find the probability of churn.

<p align='center'>
<img src='https://github.com/ShivaliPatel/Data-science-projects/blob/master/Telco_customer_churn/Images/Churn_probability_distribution.PNG'
width=400 height=350></img>
</p>

Once, we determine very high/high churn probability customers, we can apply proper retention plans.

<p align='center'>
<img src='https://github.com/ShivaliPatel/Data-science-projects/blob/master/Telco_customer_churn/Images/Existing_customer_risk_type_distribution.png'
width=400 height=250></img>
</p>


