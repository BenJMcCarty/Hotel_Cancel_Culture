# Cancel Culture in Hospitality
*Predicting hotel reservation cancellations through machine learning modeling.*

Authored by Ben McCarty

* [Email](mailto:bmccarty505@gmail.com)
* [LinkedIn](www.linkedin.com/in/bmccarty505)
* [GitHub](https://github.com/BenJMcCarty)


## Overview

A one-paragraph overview of the project, including the business problem, data, methods, results and recommendations.

* Problem: hotels need to know the likelihood of a reservation not actualizing (cancelling or DNA) for forecasting business
* Data: reservation data from two European hotels from 2015-2017
* Methods: performed machine learning modeling techniques to determine the top reservation details to indicate whether a reservation wil actualize
* Results: *pending*
* Recommendations: *pending*

## Business Problem

Summary of the business problem you are trying to solve, and the data questions that you plan to answer in order to solve them.


* What are the business's pain points related to this project?
    * Cancellations/no-shows negatively impact revenue - prevent other bookings; hard to collect on no-show reservations
    * Operations teams rely heavily on accurate forecasts for scheduling and supplies
    
* How did you pick the data analysis question(s) that you did?
    * Prior experience in industry - aim to minimize disruption to guests and hotel staff; increase revenue by identifying reservations most likely to cancel to confirm bookings in advance

* Why are these questions important from a business perspective?
    * Maximizing revenue:
        * Anticipating number of no-show reservations
        * Determining by how many rooms to oversell (assuming cancellations/no-shows)
    * Minimizing costs:
        * Minimizing labor, supply costs using accuarate occupancy forecasts
        * Addressing potential causes for cancellations (e.g. restricting number of bookings from an OTA with high likelihood of cancellations)
        * Minimizing reservation relocation costs in case of oversell


## Data

Describe the data being used for this project.

***
Questions to consider:
* Where did the data come from, and how do they relate to the data analysis questions?
    * Reservation data originally sourced from two anonymous hotels in Europe
* What do the data represent? Who is in the sample and what variables are included?
    * Each observation represents a single reservation
    * Reservation characteristics are common for most hotels (e.g. room types, marketing details, rates, and room types)
* What is the target variable?
    * Target variable is "`is_canceled`," representing whether the reservation actualized (stayed and checked-out) or if the reservation cancelled 
        * Cancellations include a small number of no-show reservations; considered to be canceled for analysis and predictions
* What are the properties of the variables you intend to use?
    * Mix of categorical and continuous data, such as room type booked/assigned; number of guests; rate; and meal type purchased with reservation.
***

## Methods

Describe the process for analyzing or modeling the data. For Phase 1, this will be descriptive analysis.

***
Questions to consider:
* How did you prepare, analyze or model the data?
    * Preparations included:
        * Addressing missing values by dropping a characteristic with 95% of the values missing
        * Filling missing values for the "agent" characteristic with a placeholder value
        * Filling in the few remaining missing entries with the most frequent values for each characteristic
    * Exploratory analysis included statistical overviews and visualizations of each characteristic's data
    * Modeling techniques utilized a logisitic regression model as well as variations of tree-based models
        * Logisitic regression results can be calculated as the odds that a 
* Why is this approach appropriate given the data and the business problem?

***

## Results

Present your key results. For Phase 1, this will be findings from your descriptive analysis.

***
Questions to consider:
* How do you interpret the results?
* How confident are you that your results would generalize beyond the data you have?
***

Here is an example of how to embed images from your sub-folder:

### Visual 1
![graph1](./images/viz1.png)

## Conclusions

Provide your conclusions about the work you've done, including any limitations or next steps.

***
Questions to consider:
* What would you recommend the business do as a result of this work?
* What are some reasons why your analysis might not fully solve the business problem?
* What else could you do in the future to improve this project?
***

## For More Information

Please review our full analysis in [our Jupyter Notebook](./dsc-phase1-project-template.ipynb) or our [presentation](./DS_Project_Presentation.pdf).

For any additional questions, please contact **name & email, name & email**

## Repository Structure

Describe the structure of your repository and its contents, for example:

```
├── README.md                           <- The top-level README for reviewers of this project
├── dsc-phase1-project-template.ipynb   <- Narrative documentation of analysis in Jupyter notebook
├── DS_Project_Presentation.pdf         <- PDF version of project presentation
├── data                                <- Both sourced externally and generated from code
└── images                              <- Both sourced externally and generated from code
```
