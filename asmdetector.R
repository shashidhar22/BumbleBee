args<-commandArgs(TRUE)
library(e1071)
set.seed(77)
var <- read.csv(args[1],header=TRUE)
var[var == 2] = 1
var <- var[ , ! apply( var , 2 , function(x) all(is.na(x)) ) ]
true <- 0
false <- 0
for (i in 1:nrow(var)){
    fit <- naiveBayes(X ~ ., data=var[-i,], na.action=na.pass)
    pred <- predict(fit,var[i,])
    if(is.na(pred[1])){
        false <- false + 1
    }
    else if(pred[1] == var[i,]$X){
        true <- true + 1
    }
    else{
    false <- false + 1
    }
}
acc <- true/(true+false)
print(acc)

