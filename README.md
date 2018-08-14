# Selected-machine-learning-methods-report-in-Markdown

In the *parkinson_ml2.R* file there is an analysis of the Parkinsons Telemonitoring Data Set downloaded from the UCI Machine Learning Repository. The *rf_report.html* file is a html report generated in RMarkdown (*rf_report.Rmd*) for the analysis performed in the *parkinson_ml2.R* file.

The analysis is based primarily on machine learning using the **mlr package**. In addition, the script contains elements of analysis using data.table, Rlof, randomForest, e1071 packages.

**The analysis includes:**
- visualization of outliers
- random forest using the mlr and randomForest packages
- SVM (Support Vector Machines) using the mlr package

**The analysis using the mlr package includes:**
- data standardization
- creating a task (regression)
- training the model (randomForest and SVM)
- tuning hyperparameters
- learner’s update
- training the model with optimal hyperparameters
- constructing Partial Dependence Plots
- testing the importance of variables
- comparison of different models (lm, randomForest, SVM) with resampling method = cross-validation

To manually generate a html file (and not use the ready one (*rf_report.html*), located in the repository), you should first perform the analysis from the *parkinson_ml2.R* file to get the .rda files necessary for the report.
Results are not generated when you create a report, because it would take too long. The data set is quite large, and some of the analyzes are based on resampling. 
