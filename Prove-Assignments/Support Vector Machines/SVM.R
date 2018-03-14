library (e1071)
library(readr)

print("Welcome to the support vector machine.")
print("Please select a dataset.")
print("1 - Letters dataset")
print("2 - Vowel dataset")

dataset <- readline("> ")

# prepare for letter and vowel data sets
gamma <- c(0.001, 0.05, 0.01, 0.5, 0.1, 1)
letter_gamma <- -1
letter_cost <- -1
letter_accuracy <- -1
letter_prediction <- -1
vowel_gamma <- 1
vowel_cost <- 1
vowel_accuracy <- -1
vowel_prediction <- -1

if (dataset == 1) {
  cat("--------------LETTERS DATASET-------------\n")
  
  fileName <- "C:/Users/bradr/R/R_prjs/CS450/Data/letters.csv"
  data <- read.csv(fileName, head=TRUE, sep=",")
  
  # Partition the data into training and test sets
  # by getting a random 30% of the rows as the testRows
  allRows <- 1:nrow(data)
  testRows <- sample(allRows, trunc(length(allRows) * 0.3))
  
  # setup the training and testing data
  train_data <- data[-testRows,]
  test_data <- data[testRows,]
  
  
  for (g in gamma) {
    cat("EXECUTING...\n")
    
    for (c in 1:2) {
      # Train an SVM model
      model <- svm(letter~., data = train_data, kernel = "radial", gamma = g, cost = c, type="C")
      
      # Use the model to make a prediction on the test dataset
      prediction <- predict(model, test_data[,-1])
      
      # Produce a confusion matrix
      confusion_matrix <- table(pred = prediction, true = test_data[,1])
      
      # Calculate the accuracy
      agreement <- prediction == test_data[,1]
      accuracy <- prop.table(table(agreement))
      num_correct_predictions <- length(which(agreement == TRUE))
      
      if (num_correct_predictions > letter_accuracy) {
        letter_cost <- c
        letter_gamma <- g
        letter_accuracy <- num_correct_predictions
        letter_prediction <- accuracy
      } else {
        1
      }
    }
  }
  
  cat("Letter Cost: ", letter_cost, "\n")
  cat("Letter gamma: ", letter_gamma, "\n")
  cat("Letter accuracy: ", letter_prediction, "\n")
} else if (dataset == 2) {
  cat("--------------VOWEL DATASET-------------\n")
  

  fileName <- "C:/Users/bradr/R/R_prjs/CS450/Data/vowel.csv"
  data <- read.csv(fileName, head=TRUE, sep=",")
  
  # Partition the data into training and test sets
  # by getting a random 30% of the rows as the testRows
  allRows <- 1:nrow(data)
  testRows <- sample(allRows, trunc(length(allRows) * 0.3))
  
  
  # set up the training and testing data
  train_data <- data[-testRows,] 
  test_data <- data[testRows,]
  
  
  for (g in gamma) {
    cat("EXECUTING...\n")
   
     for (c in 1:25) {
     
      # Train an SVM model
      model <- svm(Class~., data = train_data, kernel = "radial", gamma = g, cost = c)
      
      # Use the model to make a prediction on the test dataset
      prediction <- predict(model, test_data[,-13])
      
      # Produce a confusion matrix
      confusion_matrix <- table(pred = prediction, true = test_data[,13])
      
      # Calculate the accuracy
      agreement <- prediction == test_data[,13]
      accuracy <- prop.table(table(agreement))
      num_correct_predictions <- length(which(agreement == TRUE))
      
      if (num_correct_predictions > vowel_accuracy) {
        vowel_cost <- c
        vowel_gamma <- g
        vowel_accuracy <- num_correct_predictions
        vowel_prediction <- accuracy
      } else {
        1
      }
    }
  }
  
  cat("Vowel Cost: ", vowel_cost, "\n")
  cat("Vowel gamma: ", vowel_gamma, "\n")
  cat("Vowel accuracy: ", vowel_prediction, "\n")
} else {
  stop("ERROR! Invalid entry")
}