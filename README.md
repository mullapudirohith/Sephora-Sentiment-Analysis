## CORRELATION OF CHEMICALS AND CUSTOMER SATISFACTION AT SEPHORA
![image](https://github.com/mullapudirohith/Sephora-Sentiment-Analysis/assets/53976690/72c2ac69-127c-424c-be74-9ae92cadb2aa)



## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Models](#models)
- [Usage](#usage)
- [Contributors](#contributors)

## Introduction
To identify statistically significant correlations between the chemical components of cosmetics and customer satisfaction and to develop predictive models that estimate customer satisfaction based on product chemical compositions. We use product ingredients and customer reviews data scraped from [Sephora](www.sephora.com).

Then went on to use sentiment analysis, feature engineering and various evaluation methods for model selection – eventually reaching our ideal model. This project aims to leverage data science to enhance Sephora’s decision-making process, enabling more informed product formulation and marketing strategies that align with consumer health consciousness and
preferences.

## Features
Below is the dataframe, that we feed into the model, product_id is dropped, X was functional groups and other column. <br><br>
<p align="center">
<img src="https://github.com/mullapudirohith/Sephora-Sentiment-Analysis/assets/53976690/849c8a31-6d39-4532-b327-fbc1867d8900" alt="image">
</p>

**Feature Engineering:** <br>
* Frequency-Based Selection: <br>
Selected features based on the chemical ingredients' presence in various
cosmetic products. Specifically, chemicals that appeared in at least 15 different products. This threshold
ensured that only ingredients with a significant frequency across the dataset were included.

* Functional Group Bins:<br>
Categorized chemicals into
based on their functional groups. This classification
provided additional context on the types of
compounds present, allowing us to explore their
potential relationships with customer satisfaction.

<p align="center">
<img src="https://github.com/mullapudirohith/Sephora-Sentiment-Analysis/assets/53976690/8f25aff3-07de-4fab-bec7-6ab8e4fd15ca" alt="image" style="text-align: center; width: 600px; height: 500px;">
</p>

## Requirements
1. Python version 3.xx
2. PyCharm/Colab
3. Internet Connection
4. 16 GB RAM
5. M1 or Intel equivalent CPU
6. GPU

## Installation

Follow this step if you are running the notebook locally, to create and activate a virtual environment using venv, follow these steps:

1. **Create Virtual Environment:**

   Navigate to your project directory in the terminal and run the following command:

   ```bash
   python -m venv myen
   ```
   Replace myenv with the name you want to give to your virtual environment.

2. **Activate the Virtual Environment:On macOS/Linux:**
   ```bash
   source myenv/bin/activate
   ```
   On Windows:
   ```bash
   myenv\Scripts\activate
   ```
   You should see (myenv) or the name of your virtual environment appear at the beginning of your command prompt, indicating that the virtual environment is activated.
  
3. **Install Packages:**<br/>
     To install the required Python packages, you can use pip:
    ```bash
    pip install pandas bs4 requests jupyter gdown nltk matplotlib scikit-learn
    ```
## Models
<table align="center">
<tr><th>Regression Model </th><th>Classification Model</th></tr>
<tr><td>

| Model                   | RMSE   |
|-------------------------|--------|
| MSE Baseline            | 0.0249 |
| Linear Regression       | 0.0148 |
| Ridge Regression        | 0.0148 |
| Random Forest Regressor | 0.0148 |

</td><td>

| Model                   | Accuracy |
|-------------------------|----------|
| Baseline Classifier     | 37.7%    |
| Logistic Regression     | 53.3%    |
| Decision Tree           | 51.3%    |
| Random Forest Classifier| 54.6%    |

</td></tr> </table>




### Performance: 
<img src="https://github.com/mullapudirohith/Sephora-Sentiment-Analysis/assets/53976690/02fd278c-38ef-459a-91c3-c69cf9124220" alt="image" >
<img src="https://github.com/mullapudirohith/Sephora-Sentiment-Analysis/assets/53976690/2937b562-0758-43f4-8032-b05ca77c604b" alt="image">



## Usage
1. **Product Performance Prediction:** Our model utilizes one-hot encoded chemical features to predict which product out of two alternatives will likely perform better in terms of sales or customer satisfaction. By analyzing historical data and identifying correlations between product characteristics and performance metrics, the model assists in decision-making processes, such as inventory management and product placement strategies.

2. **Optimized Product Assortment:** Leveraging the predictive capabilities of our model, retailers can optimize their product assortment by selecting items that are more likely to resonate with customers based on chemical compositions. For example, when faced with choosing between two similar skincare products, retailers can rely on the model's insights to stock the item with ingredients that align better with customer preferences, ultimately maximizing sales potential.

3. **Enhanced Marketing Strategies:** Armed with predictive insights from our model, marketing teams can craft more targeted and effective campaigns. By understanding which product variations are more likely to appeal to specific customer segments, marketers can tailor messaging and promotions accordingly, driving engagement and conversion rates.

## Contributors
1️⃣. [Rohith Mullapudi](https://github.com/mullapudirohith)<br>
2️⃣. [Aniket Verma](https://github.com/aniketverma-7)<br>
3️⃣. [Atman Wagle](https://github.com/atmanwagle)<br>
4️⃣. [Kareena Parwani]()<br>


