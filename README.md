# Med-Care Wellness Research Center
## Team members
Minjae Hwang E00184 (“Captain")

Nina Brugge E00462

Nicole Rosas E00317
 
# Introduction
The MedCare Wellness Research Center project aims to utilize artificial intelligence in healthcare, with a specific focus on the elderly population. Given the worldwide trend of an increasingly elderly population, it is crucial to comprehend and forecast health issues that are unique to this demographic. The project employs an extensive dataset obtained from nationwide surveys, which includes a diverse range of health indicators and lifestyle characteristics of the elderly population. This initiative seeks to proactively detect potential health issues by employing sophisticated machine learning techniques to meticulously analyze patterns and indicators. This initiative aims to proactively display an overview of the physical condition of older people by employing sophisticated machine learning techniques to meticulously analyze patterns and indicators. We are dedicated to utilizing technology to improve the well-being of the elderly population, leading to a future where healthcare is more anticipatory, tailored, and proactive.

# Methods
## Data Preprocessing 
Our dataset underwent preprocessing steps, ensuring its readiness for analysis. Key steps included:

#### Handling Missing Values
Ensuring the completeness of data by verifying and validating the absence of any missing values.

#### Column Removal
The 'Patient ID' column was excluded from the analysis due to its lack of relevance, resulting in a reduction in the number of features to 18.

#### Categorical to Numerical Conversion
Employed binary, integer, and one-hot encoding techniques to convert categorical variables, hence improving the interpretability and computational efficiency of the model. The majority of our binary variables consisted of replies categorized as either 'Yes' or 'No'. In order to translate them into numerical variables, we employed binary encoding, whereby 'Yes' was converted to '1' and 'No' to '0'. Nevertheless, the binary variable 'Gender' is the only exception. For this instance, we converted the gender 'Female' to '1' and 'Male' to '0'. For the variables 'How Do You Feel' and 'Age Group', which have a hierarchical relationship, we utilized integer encoding. Ultimately, due to the absence of a binary classification or hierarchy in the 'Ethnicity' variable, we utilized one-hot encoding to transform it into a dummy variable.

## Data Visualization
The data visualization process played a critical role in our project, as it attempted to present a thorough overview of the distribution of our information and the complex relationships between different factors. This technique played a crucial role in uncovering the dynamics and relationships within our data, with a specific emphasis on 'Physical Health' as our main variable of interest.

#### Correlation heatmap
We employed a correlation heatmap, an effective tool that graphically displays the correlation coefficients among several variables. This method was highly efficient in determining the correlation between several parameters and 'Physical Health'.

There is a strong positive link (ρ ≥ 0.3) between walking difficulties and mental health, and a negative correlation between how you feel and mental health. 

Correlation within the range of 0.1 to less than 0.3: Torsades de Pointes, Do you Exercise, Asthma condition, Renal Disorder, Smoking, diabetes, age group, and body mass index Stroke's historical background

Negligible connection (ρ < 0.1) exists between Skin Cancer, Hours of Sleep, Gender, How Many Drinks per Week, and Ethnicity (all of them). 

Although the division was carried out, it is crucial to acknowledge that none of the factors exhibit a genuinely significant correlation with the intended variable. The optimal situation is to strive for a correlation that is around 1 between variables.

#### Histograms for Continuous Variables
We graphically represented the dispersion and central tendency of the variables 'Physical Health', 'Hours of Sleep', 'Drinks a Week', and 'Body Mass Index (BMI)' in order to gain insight into their distribution within the dataset.

Allocation of physical health resources: 
The histogram depicting 'Physical Health' indicates that a significant proportion of persons report experiencing minimal physical health concerns, with a sharp decrease shown as the number of health issues rises. This disparity indicates that while the majority of persons do not report serious health issues, a smaller subgroup encounters a substantial number of physical health problems.

#### Bar Charts for Categorical Variables: 
Bar charts are used to analyze categorical data. In this scenario, we are using a bar chart to determine the frequency of binary variables that have a significant or moderate impact on our desired output variable. 

Health conditions: The prevalence of health disorders such as 'Walking Difficulty', 'Torsades de Pointes', 'Asthma Status', and 'Kidney Disease' is relatively low, as indicated by the bar charts. Nevertheless, the existence of these illnesses underscores the necessity for focused healthcare interventions for individuals who are impacted.

![image](https://github.com/MinjaeHwangMo/MedCarePJ_E00184/assets/153005474/ffe54b6a-f937-46b3-bcaa-0c5bfb71f30a)

Age Group: The majority of individuals in our sample have a median age, neither very young (under 18) nor very old (over 80). 

![image](https://github.com/MinjaeHwangMo/MedCarePJ_E00184/assets/153005474/877853e9-0387-46a0-8738-b3433fdbe46c)

Lifestyle Choices: The majority of the sample are active people who exercise. However, even though a little over half of our sample does not usually smoke, we have a significant portion of our population that does.    

![image](https://github.com/MinjaeHwangMo/MedCarePJ_E00184/assets/153005474/c00db89a-d941-45df-98df-2acb9ad1344b)

It can be concluded that we have a population that mostly engages in sports. This may be due to their age, which allows them to continue engaging in physical activities. Additionally, the majority of the sample are non-smokers, which could explain why there are not many cases of people with difficulty walking or an abnormal heart rhythm. However, it may not be surprising if the health of this group of individuals declines a bit, as there is a significant portion of the population with the habit of smoking, which would likely affect their physical health. 

#### Boxplots
To analyze the distribution of continuous variables across different categories of 'How do you Feel', we employed boxplots for 'Hours of Sleep' and 'Body Mass Index (BMI)'. These plots were instrumental in identifying trends and potential outliers. In our case, the boxplots display a clear trend where individuals who feel better tend to report fewer sleep issues and maintain a healthier BMI. This suggests a correlation between self-reported health feelings and other health indicators.

#### Count Plot
We created a count plot to observe the frequency of drinking status across different self-reported health statuses, which revealed interesting patterns about lifestyle choices among various health perception groups. For our project, the count plot and normalized bar charts illustrate a correlation between lifestyle choices (like exercise and smoking) and self-reported health status. Those who exercise more or smoke less tend to report feeling better about their health.

#### Normalized Bar Charts
We normalized counts within each 'How do you Feel' category for variables such as 'Do you Exercise', 'Is Smoking', and 'Diabetes'. This allowed us to observe the proportion of individuals with these characteristics within each self-reported health category. [a]

#### Pair Plot for Continuous Variables
Finally, we used pair plots to visualize the relationships between continuous variables such as 'Hours of Sleep', 'BMI', 'Physical Health', and 'Mental Health', with 'How do you Feel' as the hue. This multivariate analysis helped to uncover complex interdependencies between different health indicators. For our case, the pair plots reveal complex relationships between continuous variables. For instance, 'Physical Health' and 'Mental Health' appear correlated, indicating that individuals who report poor physical health also tend to report poor mental health.

These visualizations imply that while the dataset contains a healthy segment of the population, there are clear indications of health issues correlated with both lifestyle choices and self-reported feelings of health. The correlations found could be used to guide further analysis, predictive modeling, and targeted health interventions.

# Experimental Design
In this project, we conducted a series of experiments to improve the prediction of 'Physical Health' using different modeling techniques. Starting with a simple Multiple Linear Regression (MLR) model to set our baseline, we then explored more complex models like Random Forest and Ensemble Learning.

## Multiple Linear Regression (MLR) 

#### Main purpose
In our first experiment, we employed the Multiple Linear Regression (MLR) model to explore how various factors linearly relate to 'Physical Health', setting the groundwork for predicting accuracy. As our initial approach, MLR served as the baseline, providing a fundamental comparison point to gauge the advancements made by more complex models in subsequent experiments.
#### Baseline
As the initial model, MLR itself is the baseline in this context. It provides a standard against which the complexity and effectiveness of subsequent models can be compared.
#### Evaluation Metrics
We utilized R-squared to determine how much variance in 'Physical Health' the MLR model could explain, as well as Mean Squared Error (MSE) and Mean Absolute Error (MAE) to examine prediction mistakes. These measures were selected because they provide a clear and succinct picture of the model's predicted accuracy and error magnitude.

## Random Forest 

#### Main Purpose
To improve on the predictive accuracy provided by the MLR model by using a Random Forest model to capture non-linear patterns and interactions in the data.
#### Baseline
The performance of the MLR model served as the experiment's baseline. The purpose of the comparison was to determine the added value of employing a more complicated and sophisticated model.
#### Evaluation Metrics
R-squared, MSE, and MAE were used as evaluation metrics, as was the MLR model. We were able to directly compare the Random Forest model's performance to the MLR baseline using these metrics, identifying areas for improvement or the need for further model modification. But emphasized the role of MSE and MAE in highlighting Random Forest's ability to manage prediction errors.

## Ensemble Learning 
#### Main Purpose
The main goal of this experiment was to construct a more robust predictive model by merging different models, including RandomForest and GradientBoosting. The goal was to outperform individual models by using their combined strengths.
#### Baseline
The MLR and Random Forest models' performances served as the experiment's baselines. This comparison assisted in comprehending the incremental advantages of merging multiple models.
#### Evaluation Metrics
We continued with the same set of metrics: R-squared, MSE, and MAE. These metrics were instrumental in demonstrating how the Ensemble method's predictions were more accurate or less error-prone compared to the individual models of MLR and Random Forest. We specifically focused on these metrics to illustrate the Ensemble model's superior capacity in minimizing prediction errors and enhancing overall accuracy.

# Results
## R-squared 
The R-squared values obtained from our tests revealed interesting patterns.

The Multiple Linear Regression (MLR) model exhibited the greatest R-squared value of 0.33, indicating that it accounted for approximately 33% of the variability in 'Physical Health'. This indicates that although MLR yielded a satisfactory level of accuracy, there is still a substantial amount of variability that cannot be accounted for.  

On the other hand, the Random Forest and Ensemble Learning models produced lower R-squared values (0.05 and 0.11, respectively), suggesting that they were less successful in capturing the variability of the target variable. This result implies that, despite their intricate nature, these models may not accurately correspond to the structure of the dataset or adequately tackle its inherent intricacies.

![image](https://github.com/MinjaeHwangMo/MedCarePJ_E00184/assets/153005474/2a66c99c-726c-42a1-bbe8-2c50843e3c6c)

## Mean Squared Error (MSE)
Regarding the Mean Squared Error (MSE), our investigation produced the following findings:

The Multiple Linear Regression (MLR) model had the lowest Mean Squared Error (MSE) value of 43.82. This indicates that, on average, the squared prediction errors for MLR were less in comparison to the other models, indicating a more accurate alignment with the actual data.

In contrast, the Random Forest and Ensemble Learning models demonstrated higher Mean Squared Error (MSE) values, specifically 61.84 and 57.97 respectively. The higher numbers suggest that these models, on average, had larger squared errors in their predictions. This outcome suggests that these models may exhibit worse accuracy in predicting 'Physical Health' when compared to MLR.  

The variability in Mean Squared Error (MSE) values among various models offers useful insights into their predictive precision, suggesting a possible compromise between model intricacy and prediction mistake.

![image](https://github.com/MinjaeHwangMo/MedCarePJ_E00184/assets/153005474/b8839b41-d8c2-495a-865c-5089e7f24f7f)

## Root Mean Squared Error (RMSE)
The Root Mean Squared Error (RMSE) results from our models yielded the following insights:

The Multiple Linear Regression (MLR) model attained the minimum Root Mean Square Error (RMSE) of 6.62. This suggests that the MLR model has a lesser average error in predictions, when assessed in the same units as the 'Physical Health' variable. It indicates a comparatively more accurate alignment with the real facts.

Conversely, the Random Forest and Ensemble Learning models had higher RMSE values, specifically 7.86 and 7.61 respectively. These higher numbers indicate that, on average, these models made more significant errors in their forecasts. This may indicate a lower level of accuracy in these models in comparison to the MLR model.  

The disparity in RMSE across the models offers valuable insights into the precision of each model's predictions, with smaller values indicating a higher degree of proximity to the actual data.

![image](https://github.com/MinjaeHwangMo/MedCarePJ_E00184/assets/153005474/d4b32ac9-9a55-4075-9077-20085b759a90)

## Mean Absolute Error (MAE)
The models yielded the Mean Absolute Error (MAE) as the outcome, which revealed the following significant findings:

The Ensemble Learning model demonstrated the lowest Mean Absolute Error (MAE) of 2.84, suggesting that its predictions had, on average, less absolute errors. These findings indicate that the Ensemble model had the highest level of accuracy when measuring the average deviation from the real 'Physical Health' values.

By comparison, the Multiple Linear Regression (MLR) and Random Forest models exhibited greater Mean Absolute Error (MAE) values, specifically 4.18 and 3.09 respectively. The larger Mean Absolute Error (MAE) values indicate that these models had higher average absolute errors in their predictions.  

The MAE measure indicates that the Ensemble Learning model has a more accurate predictive capability, with a lower average error size compared to MLR and Random Forest.

Overall, Our research of the performance measures indicates that the Multiple Linear Regression (MLR) model demonstrates superior performance compared to both the Random Forest and Ensemble Learning models in this particular situation. This conclusion is based on the observation that MLR had the highest R-squared value, together with the lowest Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). The findings indicate that MLR not only accounts for a larger amount of the variability in 'Physical Health', but also exhibits lower average errors and variation in its predictions compared to the more intricate models. This result demonstrates the efficacy of MLR in terms of its ability to provide both a comprehensive explanation and accurate predictions for this specific dataset and prediction job.

![image](https://github.com/MinjaeHwangMo/MedCarePJ_E00184/assets/153005474/77a7b967-ca7d-413c-8851-460be71bfd01)

# Conclusion
Our research into predicting 'Physical Health' using machine learning models generated interesting results. While simple, the Multiple Linear Regression model performed very well in capturing the variance of health scores. However, as measured by the RMSE and MAE measures, this did not result in the lowest prediction errors. Ensemble Learning approaches, which incorporated the capabilities of many models, demonstrated a reduced MAE, implying a potentially more consistent and robust prediction in real-world scenarios. The heterogeneity in model performance across variables highlights the complex character of predictive modeling, in which no single model excels on all fronts.

### Unanswered Questions and Future Work
While our models are a promising first step toward predictive health analytics, a number of questions remain. The complexities of 'Physical Health' as a variable raise concerns regarding undiscovered complexities within the predictors that our models may have missed. For example, the relationships between various health disorders and lifestyle choices could be investigated further. Moving forward, we advise for progress through advanced feature engineering and non-linear model exploration to more effectively capture the complexity of health data. To ensure broad relevance and accuracy, additional research might look into the applicability of our models across diverse demographics.
