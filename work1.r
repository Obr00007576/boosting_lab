library(mlbench)
library(adabag)

data(Vehicle)

set.seed(333)
train <- sample(nrow(Vehicle), 0.7 * nrow(Vehicle))
train_data <- Vehicle[train, ]
test_data <- Vehicle[-train, ]

# Train an AdaBoost.M1 model with adabag
n_trees <- seq(1, 301, by = 10)

error_list <- list()
for(n in n_trees){
    ada_model <- boosting(Class ~ ., data = train_data, mfinal = n)
    test_pred <- predict.boosting(ada_model, newdata = test_data)
    test_error <- mean(test_pred$class != test_data$Class)
    error_list[[as.character(n)]] <- test_error
}

# Draw Test error as a function of Number of trees
plot(n_trees, sapply(error_list, unlist), type = "b", xlab = "Number of trees", ylab = "Test error")
title("Vehicle")