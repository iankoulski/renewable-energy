# KubeCon 2020 Demo (renewable-energy)

This repository is built as an example of a typical data science project. Its intended use is to demonstrate the data scientist's workflow when turning the project into a Kubeflow pipeline as part of the following presentation at KubeCon 2020: [https://sched.co/ekBt](https://sched.co/ekBt)

## Problem Statement
The state of California has a goal to generate 60% of its electrical energy from renewable sources by 2030 and carbon-neutral by 2045. The California Independent System Operator (CAISO) provides a daily bulletin where it publicly discloses energy generation data for the state's grid broken down by source and recorded on an hourly basis.

With this project we'd like to model the growth of renewable energy using the available historical data, then project future growth and estimate the likelihood of CA reaching its renewable energy goals.

```
   Disclaimer: 
      The CAISO reports use raw data and are not intended
      to be used as the basis for operational or financial decisions.
```

## Solution Approach
The solution approach is to break down the problem and solve of the following subproblems

### Data wrangling
Download all available data from the bulletins published on CAISO's website

### Data preprocessing
Combine all data into a single table, assess data quality and perform clensing if needed. Split the data into a training and validation set.

### Modeling technique selection
Build several models using different modeling techniques and the training dataset, validate the models and compute model quality metrics based on the validation set. Select the best model

### Model building
Train a model using the full dataset and the selected best modeling technique.

### Predict 
Predict renewable energy generation by 2030 and 2045 with confidence ranges. Calculate likelihood of achieving goals.

## Data Science Workflow

The data sicentist develops the project using python locally at first, then moves to a cloud environment to accelerate model training, and finally uses the Kale SDK to annotate the code and turn it into a Kubeflow pipeline.  

## References

* power-technology.com Article: [https://www.power-technology.com/features/california-renewables-on-the-frontline](https://www.power-technology.com/features/california-renewables-on-the-frontline)
* CAISO renewables reports: [http://www.caiso.com/market/Pages/ReportsBulletins/RenewablesReporting.aspx](http://www.caiso.com/market/Pages/ReportsBulletins/RenewablesReporting.aspx)
* Kubeflow:  [https://kubeflow.org](https://kubeflow.org)
* Kale: [https://github.com/kubeflow-kale/kale](https://github.com/kubeflow-kale/kale)
* Kale SDK: [https://docs.arrikto.com/user_guides/kale.html](https://docs.arrikto.com/user_guides/kale.html)
* Arrikto: [https://www.arrikto.com/](https://www.arrikto.com/)