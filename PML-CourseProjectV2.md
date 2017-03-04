Practical Machine Learning  - Final course project
================================================================
## using Random Forest package 

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).


```r
library(knitr)
opts_chunk$set(warning = FALSE, cache=TRUE, fig.align='center', fig.show='asis')
library(RCurl)
library(caret)
library(dplyr)
library(randomForest)
library(ElemStatLearn)


trainlink       <- getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
dftrain         <- read.csv(text = trainlink, na.strings = "NA")
testlink        <- getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
dftest          <- read.csv(text = testlink, na.strings = "NA")
```


```r
#
#       Compare column names - should be all the same except for outcome column classe
#
"classe" %in% names(dftrain)
```

```
## [1] TRUE
```

```r
"classe" %in% names(dftest)
```

```
## [1] FALSE
```

```r
dfnames         <- data.frame(trn=names(dftrain), tsn=names(dftest))
dfnames[as.character(dfnames$trn) != as.character(dfnames$tsn),]
```

```
##        trn        tsn
## 160 classe problem_id
```
###     Cleanup - remove unnecessary columns

```r
#       cleanup - some columns have many NAs - check first
#               check actual row counts for NA values - get %
arnas <- colSums(is.na(dftrain))/nrow(dftrain)*100
arnas[arnas>0]  
```

```
##            max_roll_belt           max_picth_belt            min_roll_belt 
##                 97.93089                 97.93089                 97.93089 
##           min_pitch_belt      amplitude_roll_belt     amplitude_pitch_belt 
##                 97.93089                 97.93089                 97.93089 
##     var_total_accel_belt            avg_roll_belt         stddev_roll_belt 
##                 97.93089                 97.93089                 97.93089 
##            var_roll_belt           avg_pitch_belt        stddev_pitch_belt 
##                 97.93089                 97.93089                 97.93089 
##           var_pitch_belt             avg_yaw_belt          stddev_yaw_belt 
##                 97.93089                 97.93089                 97.93089 
##             var_yaw_belt            var_accel_arm             avg_roll_arm 
##                 97.93089                 97.93089                 97.93089 
##          stddev_roll_arm             var_roll_arm            avg_pitch_arm 
##                 97.93089                 97.93089                 97.93089 
##         stddev_pitch_arm            var_pitch_arm              avg_yaw_arm 
##                 97.93089                 97.93089                 97.93089 
##           stddev_yaw_arm              var_yaw_arm             max_roll_arm 
##                 97.93089                 97.93089                 97.93089 
##            max_picth_arm              max_yaw_arm             min_roll_arm 
##                 97.93089                 97.93089                 97.93089 
##            min_pitch_arm              min_yaw_arm       amplitude_roll_arm 
##                 97.93089                 97.93089                 97.93089 
##      amplitude_pitch_arm        amplitude_yaw_arm        max_roll_dumbbell 
##                 97.93089                 97.93089                 97.93089 
##       max_picth_dumbbell        min_roll_dumbbell       min_pitch_dumbbell 
##                 97.93089                 97.93089                 97.93089 
##  amplitude_roll_dumbbell amplitude_pitch_dumbbell       var_accel_dumbbell 
##                 97.93089                 97.93089                 97.93089 
##        avg_roll_dumbbell     stddev_roll_dumbbell        var_roll_dumbbell 
##                 97.93089                 97.93089                 97.93089 
##       avg_pitch_dumbbell    stddev_pitch_dumbbell       var_pitch_dumbbell 
##                 97.93089                 97.93089                 97.93089 
##         avg_yaw_dumbbell      stddev_yaw_dumbbell         var_yaw_dumbbell 
##                 97.93089                 97.93089                 97.93089 
##         max_roll_forearm        max_picth_forearm         min_roll_forearm 
##                 97.93089                 97.93089                 97.93089 
##        min_pitch_forearm   amplitude_roll_forearm  amplitude_pitch_forearm 
##                 97.93089                 97.93089                 97.93089 
##        var_accel_forearm         avg_roll_forearm      stddev_roll_forearm 
##                 97.93089                 97.93089                 97.93089 
##         var_roll_forearm        avg_pitch_forearm     stddev_pitch_forearm 
##                 97.93089                 97.93089                 97.93089 
##        var_pitch_forearm          avg_yaw_forearm       stddev_yaw_forearm 
##                 97.93089                 97.93089                 97.93089 
##          var_yaw_forearm 
##                 97.93089
```

```r
unique(arnas)           
```

```
## [1]  0.00000 97.93089
```

```r
#       so a lot of columns have 90%+ NAs ... lets remove them
#       keep columns with < 90% NAs
dftrainnona     <- dftrain[ , colSums(is.na(dftrain)) < 0.9*nrow(dftrain) ]
# ------------------ removing some additional columns due to different reasons ------------------------------
#
#       validate if there are any columns to be removed as low-variance that could make the model unstable - 
#       per this link: https://topepo.github.io/caret/pre-processing.html 

dftrainnona2     <- subset( dftrainnona, select = -c(X, user_name
                        , raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp) )


#       low variance columns - check what we get first
#nzv <- nearZeroVar(dftrainnona2, saveMetrics=TRUE)
#
#       then remove the one column that shows up
nzv             <- nearZeroVar(dftrainnona2)
dftrainnona3    <- dftrainnona2[,-nzv]

#
#       identify highly correlated columns
#
#       since there are categorical columns, convert temporary into integers to be able to use cor
df3temp         <- dftrainnona3
#       determine factor columns first, then convert to int in the df3temp
fcnames         <- names(df3temp)[sapply(df3temp, class)=="factor"]
fdf             <- df3temp[,fcnames]
intfdf <-data.frame(sapply(fdf,as.integer))
#       next gives us data frame with integeres instead of factors so cor could be done 
df4temp         <- df3temp[, !(colnames(df3temp) %in% fcnames)]  %>% bind_cols( intfdf )
corx            <- cor(df4temp)
hlyCorDescr     <- findCorrelation(corx, cutoff = .90)
#
#       here the highly correlated columns are in hlyCorDescr; remove them from the original
#       dftrainnona3 frame, and create new frame to use further;
#       NOTE: remove all highly correlated column except classe
hlyCorDescr     <- hlyCorDescr[!(hlyCorDescr %in% (which( colnames(df4temp)=="classe") ))]
dftrainnona4    <- dftrainnona3[,-hlyCorDescr]

# ------------------ using dftrainnona4 going forward --------------------------------------------------------

#
#       subset the test set the same way, handling difference in classes vs. problem_id, then cross-check 
#       all left columns again
keepcollist     <- c(unlist(names(dftrainnona4))[!(unlist(names(dftrainnona4)) 
                                                   %in% "classe")], "problem_id")
dftestnona      <- subset(dftest, select=keepcollist)
dfnames         <- data.frame(trn=names(dftrainnona4), tsn=names(dftestnona))
dfnames[as.character(dfnames$trn) != as.character(dfnames$tsn),]
```

```
##       trn        tsn
## 47 classe problem_id
```


###      set the data, split into train and test, and create 10 subsets for that purpose

```r
n_train_subset          <- 10
pct_traintotest         <- 0.8
pct_use_for_sample      <- 0.1


predf                   <- setNames( data.frame( matrix( ncol=(nrow(dftestnona)+1), nrow=0))
                                     , c( sapply(seq(1,nrow(dftestnona),1)
                                        , FUN = function(x) { paste0("c", toString(x)) } ), "accr" ) )
df1row                  <- predf
finalpred               <- predf
proc_one_set            <- function( klist, dfin, resulttest, traceall = FALSE ) {
        ps <- paste0(" ...starting proc_one_set ... time is ", toString(Sys.time()))
        cat(ps, "\n")

        #browser()
        set.seed(33833)
        sublen          <- floor(pct_use_for_sample*length(klist))
        smalllist       <- sample( klist, sublen)
        inset           <- dfin[smalllist,]
        inTrain         <- createDataPartition( y = inset$classe, p=pct_traintotest, list=FALSE )
        training        <- inset[inTrain,]
        testing         <- inset[-inTrain,]
        #

        modelFit        <- randomForest(classe~., data=training)
        if ( traceall ) { 
                plot(modelFit)
                print(modelFit,digits=3) 
        }
        #
        #       model is in modelFit$finalModel
        test4pred       <- testing[,!(names(training) %in% c("problem_id"))]
        testpred        <- predict( modelFit, test4pred )
        cfm             <- confusionMatrix(testpred, testing[,"classe"])
        if ( traceall) print(cfm, digits=4)
        acr             <- cfm$overall[1]
        ps <- paste0(" ......... completed proc_one_set with accuracy ", toString(acr) )
        cat(ps, "\n")
        #
        #       run model against final class test set
        finalpred       <- predict( modelFit, resulttest )
        for ( i in 1:(ncol(predf)-1) ) {
                df1row[1,i]     <- as.character(finalpred[i])
        }
        df1row[1,ncol(predf)]   <- acr
        predf           <<- predf %>% bind_rows( df1row )
        return ( acr )
}


folds           <- createResample( y=dftrainnona4$classe, times=n_train_subset, list=TRUE)
modsum          <- lapply(folds, proc_one_set, dfin = dftrainnona4, resulttest = dftestnona  )
```

```
##  ...starting proc_one_set ... time is 2017-03-03 16:41:03 
##  ......... completed proc_one_set with accuracy 0.969230769230769 
##  ...starting proc_one_set ... time is 2017-03-03 16:41:06 
##  ......... completed proc_one_set with accuracy 0.958974358974359 
##  ...starting proc_one_set ... time is 2017-03-03 16:41:09 
##  ......... completed proc_one_set with accuracy 0.966666666666667 
##  ...starting proc_one_set ... time is 2017-03-03 16:41:12 
##  ......... completed proc_one_set with accuracy 0.979487179487179 
##  ...starting proc_one_set ... time is 2017-03-03 16:41:14 
##  ......... completed proc_one_set with accuracy 0.964102564102564 
##  ...starting proc_one_set ... time is 2017-03-03 16:41:17 
##  ......... completed proc_one_set with accuracy 0.954081632653061 
##  ...starting proc_one_set ... time is 2017-03-03 16:41:19 
##  ......... completed proc_one_set with accuracy 0.964194373401535 
##  ...starting proc_one_set ... time is 2017-03-03 16:41:22 
##  ......... completed proc_one_set with accuracy 0.956298200514139 
##  ...starting proc_one_set ... time is 2017-03-03 16:41:24 
##  ......... completed proc_one_set with accuracy 0.964102564102564 
##  ...starting proc_one_set ... time is 2017-03-03 16:41:27 
##  ......... completed proc_one_set with accuracy 0.936061381074169
```

```r
#       get the best prediction and submit that for the quiz 
predf[predf$accr==max(predf$accr),]
```

```
##   c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19 c20
## 4  B  A  A  A  A  E  D  B  A   A   B   C   B   A   E   E   A   B   B   B
##        accr
## 4 0.9794872
```

```r
#       rerun the best run to show prints for the submisison of the project
cat("\n\n ------------------------ Final run trace data ----------------------------- \n")
```

```
## 
## 
##  ------------------------ Final run trace data -----------------------------
```

```r
bestrun         <- which(predf$accr==max(predf$accr))
predf           <- finalpred
acrfinal        <- proc_one_set( folds[[bestrun]], dfin = dftrainnona4, resulttest = dftestnona, traceall = TRUE)
```

```
##  ...starting proc_one_set ... time is 2017-03-03 16:41:30
```

<img src="figure/unnamed-chunk-4-1.png" title="plot of chunk unnamed-chunk-4" alt="plot of chunk unnamed-chunk-4" style="display: block; margin: auto;" />

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 6
## 
##         OOB estimate of  error rate: 4.58%
## Confusion matrix:
##     A   B   C   D   E class.error
## A 428   1   0   2   0 0.006960557
## B  14 285  13   1   2 0.095238095
## C   0  11 249   1   1 0.049618321
## D   2   0  13 266   0 0.053380783
## E   0   2   4   5 272 0.038869258
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 107   2   0   0   0
##          B   0  74   2   0   0
##          C   0   1  63   0   1
##          D   0   1   0  70   1
##          E   0   0   0   0  68
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9795        
##                  95% CI : (0.96, 0.9911)
##     No Information Rate : 0.2744        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9741        
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9487   0.9692   1.0000   0.9714
## Specificity            0.9929   0.9936   0.9938   0.9938   1.0000
## Pos Pred Value         0.9817   0.9737   0.9692   0.9722   1.0000
## Neg Pred Value         1.0000   0.9873   0.9938   1.0000   0.9938
## Prevalence             0.2744   0.2000   0.1667   0.1795   0.1795
## Detection Rate         0.2744   0.1897   0.1615   0.1795   0.1744
## Detection Prevalence   0.2795   0.1949   0.1667   0.1846   0.1744
## Balanced Accuracy      0.9965   0.9712   0.9815   0.9969   0.9857
##  ......... completed proc_one_set with accuracy 0.979487179487179
```
###     final submission

```r
predf[predf$accr==max(predf$accr),]
```

```
##   c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19 c20
## 1  B  A  A  A  A  E  D  B  A   A   B   C   B   A   E   E   A   B   B   B
##        accr
## 1 0.9794872
```

```r
outofsampleerror        <- 1.0 - mean(predf$accr)
```
###     out of sample error estimated at:

```r
outofsampleerror
```

```
## [1] 0.02051282
```




