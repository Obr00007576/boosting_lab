library(mlbench)
library(adabag)

data(Glass)

set.seed(333)
train <- sample(nrow(Glass), 0.7 * nrow(Glass))
train_data <- Glass[train, ]
test_data <- Glass[-train, ]

# Train an AdaBoost.M1 model with adabag
n_trees <- seq(1, 200, by = 10)

error_list <- list()
for(n in n_trees){
    ada_model <- boosting(Type ~ ., data = train_data, mfinal = n)
    test_pred <- predict.boosting(ada_model, newdata = test_data)
    test_error <- mean(test_pred$class != test_data$Type)
    error_list[[as.character(n)]] <- test_error
}

# Draw Test error as a function of Number of trees
plot(n_trees, sapply(error_list, unlist), type = "b", xlab = "Number of trees", ylab = "Test error")
title("Glass")