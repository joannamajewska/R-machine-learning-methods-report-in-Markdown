library(data.table)
library(mlr)
library(Rlof)
library(randomForest)
library(mmpf)
library(e1071)

dataset <- read.csv("C:/Users/Admin/Desktop/rf_report/parkinsons_updrs.csv", header = TRUE)
dataset$sex <- as.factor(dataset$sex)
dataset$subject. <- as.factor(dataset$subject.)
dataset <- as.data.table(dataset)
sumcol <- summarizeColumns(dataset)
mean_4subject <- dataset[, lapply(.SD, mean), by = .(subject.), 
                         .SDcols = c("test_time", "motor_UPDRS", "total_UPDRS", "Jitter...", "Shimmer", "NHR", "HNR", "RPDE", "DFA", "PPE")
                         ][order(-total_UPDRS)
                           ][test_time > 90]

# outliers ----------------------------------------------------------------

dataset2 <- dataset[, c("NHR", "HNR", "RPDE", "DFA", "PPE")]
outlier_scores <- lof(dataset2, k = 20)
outliers <- order(outlier_scores, decreasing = T)[1:10]
print(outliers)
plot(density(outlier_scores))
labels <- 1:nrow(dataset2)
labels[-outliers] <- "."
biplot(prcomp(dataset2), cex = .8, xlabs = labels)
pch <- rep(".", nrow(dataset2))
pch[outliers] <- "+"
col <- rep("black", nrow(dataset2))
col[outliers] <- "red"
pairs(dataset2, pch = pch, col = col)

# random forest -----------------------------------------------------------

dataset <- as.data.frame(dataset)
dataset <- normalizeFeatures(dataset, target = "total_UPDRS")
park_task <- makeRegrTask(id = "parkinson", 
                         data = dataset,
                         target = "total_UPDRS",
                         fixup.data = "no", 
                         check.data = TRUE)
park_rf <- train("regr.randomForest", park_task)
fitted <- predict(park_rf, park_task)
performance(fitted) 
rflearner <- makeLearner("regr.randomForest")
rfsetparam <- makeParamSet(
  makeDiscreteParam("mtry", values = 1:5), 
  makeDiscreteParam("ntree", values = seq(500, 1500, 200))
)
rftuneparam <- tuneParams(rflearner, 
                              resampling = cv3, 
                              task = park_task, 
                              par.set = rfsetparam, 
                              control = makeTuneControlGrid())
rftuneparam
save(rftuneparam, file = "rftuneparam.rda")

rftune_data <- generateHyperParsEffectData(rftuneparam)
plotHyperParsEffect(rftune_data, x = "ntree", y = "mtry", 
                    z = "mse.test.mean",
                    plot.type = "heatmap")
msetime <- as.data.frame(rftuneparam$opt.path)
rflearner <- setHyperPars(rflearner, par.vals = rftuneparam$x)
final_model <- train(rflearner, park_task)
save(park_task, file = "park_task.rda")
save(final_model, file = "final_model.rda")
performance(predict(final_model, park_task))
plotResiduals(predict(final_model, park_task))
plotResiduals(predict(final_model, park_task), type = "hist")
pdp <- generatePartialDependenceData(final_model, park_task)
save(pdp, file = "pdp.rda")
plotPartialDependence(pdp)
imp_values <- generateFilterValuesData(park_task, method = "cforest.importance")
save(imp_values, file = "imp_values.rda")
plotFilterValues(imp_values)

# appendix ----------------------------------------------------------------

filtered_ptf <- filterFeatures(task = park_task,
                                     fval = imp_values,
                                     abs = 7) 
filter_lrn <- makeFeatSelWrapper("regr.randomForest",
                                resampling = cv3,
                                control = makeFeatSelControlRandom(max.features = 7),
                                show.info = TRUE)
filter_pt <- train(filter_lrn, park_task)

# comparison of methods ---------------------------------------------------

park_benchmark <- benchmark(makeLearners(c("lm", "randomForest", "svm"),
                                         type = "regr"),
                            resamplings = cv10, 
                            park_task)
park_benchmark
save(park_benchmark, file = "park_benchmark.rda")
plotBMRSummary(park_benchmark)
plotBMRBoxplots(park_benchmark)

# support vector machines -------------------------------------------------

svmlearner <- makeLearner("regr.svm")
svmsetparam <- makeParamSet(
  makeNumericParam("cost", lower = -5, upper = 5,
                   trafo = function(x) 2^x), 
  makeNumericParam("gamma", lower = -5, upper = 5,
                   trafo = function(x) 2^x))
svmtuneparam <- tuneParams(svmlearner, 
                           resampling = cv3, 
                           task = park_task, 
                           par.set = svmsetparam, 
                           control = makeTuneControlGrid()) 
svmtuneparam
save(svmtuneparam, file = "svmtuneparam.rda")
svmtune_data <- generateHyperParsEffectData(svmtuneparam)
plotHyperParsEffect(svmtune_data, x = "mse.test.mean", y = "iteration")
plotHyperParsEffect(svmtune_data, x = "cost", y = "gamma", 
                    z = "mse.test.mean",
                    plot.type = "heatmap")
msetime2 <- as.data.frame(svmtuneparam$opt.path, trafo = TRUE)
msetime2[1:20, ]
svmlearner <- setHyperPars(svmlearner, par.vals = svmtuneparam$x)
final_model2 <- train(svmlearner, park_task)
save(final_model2, file = "final_model2.rda")
performance(predict(final_model2, park_task))
plotResiduals(predict(final_model2, park_task))
plotResiduals(predict(final_model2, park_task), type = "hist")
pdp2 <- generatePartialDependenceData(final_model2, park_task)
save(pdp2, file = "pdp2.rda")
plotPartialDependence(pdp2)
summ <- data.frame(method = c("randomForest", "svm"),
                   MSE = c(performance(predict(final_model, park_task)),
                           performance(predict(final_model2, park_task))))

# random forest without mlr -----------------------------------------------

ind <- sample(1:nrow(dataset), floor(0.7*nrow(dataset)))
trainset <- dataset[ind, ]
testset <- dataset[-ind, ]
save(trainset, file = "trainset.rda")
save(testset, file = "testset.rda")
rf <- randomForest(total_UPDRS ~., data = trainset, ntree = 500, do.trace = 25)
save(rf, file = "rf.rda")
importance(rf)
varImpPlot(rf) 
pred <- predict(rf, testset, type = "class")
cor(pred, testset$total_UPDRS)
mean(abs(testset$total_UPDRS - pred))
tune.out = tune.randomForest(trainset[, -6], trainset[, 6], ntree = c(300, 400, 500))
save(tune.out, file = "tune_out.rda")
summary(tune.out)
plot(tune.out)
