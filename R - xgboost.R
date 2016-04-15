

require (caret)
  
getwd()

mywd <- "/home/sirorezka/python_proj/Santander Customer Satisfaction"
setwd (mywd)

train = read.csv("Data/train.csv", header = TRUE)
test = read.csv("Data/test.csv", header = TRUE)

test.id <- test$ID
response <- train$TARGET 
train <- train[-c(1,371)]  ## removing columns 'id' and 'target'
test  <- test[-c(1)]

#Remove no variance predictors 
zero_var <- nearZeroVar(train, names=TRUE, freqCut = 95/5,uniqueCut = 10,saveMetrics = TRUE) 
train    <- train[,-which(zero_var$zeroVar)]
test    <- test[,-which(zero_var$zeroVar)]

train_cat_names <- list() 
train_num_names <- list()


#loop through training data by column / predictor variable 

for (i in (1:length(train))){
  if (all(train[,c(i)] == floor(train[,c(i)])))
      { train_cat_names[length(train_cat_names)+1] = (names(train[c(i)])) }
  else{ train_num_names[length(train_num_names)+1]=(names(train[c(i)])) } 
}

        
idx <- match(train_cat_names, names(train)) 
train_cat = train[,idx] 
test_cat  = test[,idx]
train_num = train[,-idx] 
test_num  = test[,-idx]        
        
#change categorical variables to factors 

for (j in (1:length(train_cat))){ 
  
  all_cat <- as.factor(c(train_cat[,c(j)],test_cat[,c(j)]))
  train_cat[,c(j)] = all_cat[1:nrow(train_cat)]
  test_cat[,c(j)] = all_cat[-(1:nrow(train_cat))]
}       


#normalize continuous variables 
        
preproc = preProcess(train_num,method = c("center", "scale")) 
train_num_standardized <- predict(preproc, train_num) 
train_standardized = cbind(train_num_standardized,train_cat,response) 
  
test_num_standardized <- predict(preproc, test_num) 
test_standardized = cbind(test_num_standardized,test_cat) 


train_standardized$response

## creating sparse data
require(xgboost)
require(Matrix)
train.y <- train_standardized$response
## train$TARGET <- NULL 
## train$TARGET <- train.y 
train_new <- sparse.model.matrix(response ~ ., data = train_standardized) 
dtrain <- xgb.DMatrix(data=train_new, label=train.y) 
watchlist <- list(train=dtrain) 



param <- list( 
  objective = "binary:logistic", 
  booster = "gbtree", 
  eval_metric = "auc", # maximizing for auc 
  eta = 0.02, # learning rate - Number of Trees 
  max_depth = 5, # maximum depth of a tree 
  subsample = .9, # subsample ratio of the training instance 
  colsample_bytree = .87, # subsample ratio of columns 
  min_child_weight = 1, # minimum sum of instance weight (defualt) 
  scale_pos_weight = 1 # helps convergance bc dataset is unbalanced 
  ) 
  
xgb <- xgb.train( 
  params = param, 
  data = dtrain, 
  nrounds = 750, 
  verbose = 1, 
  watchlist = watchlist, 
  maximize = FALSE ) 


 

newtest <- test_standardized
newtest$TARGET <- -1 
test <- sparse.model.matrix(TARGET ~ ., data = newtest) 
preds <- predict(xgb, test) 
min(preds)

result <- as.data.frame(cbind(test.id,preds))
names(result) <- c("ID", "TARGET")
write.csv(result,"predictions/R - xgboost.csv", row.names = FALSE,quote = FALSE)
