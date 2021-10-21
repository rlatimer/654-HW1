
#Task 1.1: Import tweet data
tweet_data <- read.csv('https://raw.githubusercontent.com/uo-datasci-specialization/c4-ml-fall-2021/main/data/tweet_sub.csv')

#Task 1.2
#pull out day of week
tweet_data$day <- substr(tweet_data$time,1,3)
#convert to  numeric (Monday = 1, Sunday = 7)
require(dplyr)
tweet_data$day <- dplyr::recode(tweet_data$day,
                                'Mon'=1,
                                'Tue'=2,
                                'Wed'=3,
                                'Thu'=4,
                                'Fri'=5,
                                'Sat'=6,
                                'Sun'=7)

#pull out month, convert to factor variable
tweet_data$month <- as.factor(substr(tweet_data$time,5,7))

#pull out date, convert to numeric
tweet_data$date <- as.numeric(substr(tweet_data$time,9,10))

#pull out hour, convert to numeric
tweet_data$hour <- as.numeric(substr(tweet_data$time,12,13))

#printing frequencies of new variables
#probably a more elegant way to do this but this works for now
table(day)
table(month)
table(date)
table(hour)


#Task1.3
#Recode the outcome variable (sentiment) into a binary variable such that 
#Positive is equal to 1 and Negative is equal to 0. 
#Calculate and print the frequencies for tweets with positive and negative sentiments.

tweet_data$sentiment <- dplyr::recode(tweet_data$sentiment,
                                      'Positive'=1,
                                      'Negative'=0)
table(tweet_data$sentiment)

#Task1.4
#loading packages and Python libraries
require(reticulate)

reticulate::import('torch')
reticulate::import('numpy')
reticulate::import('transformers')
reticulate::import('nltk')
reticulate::import('tokenizers')

require(text)

#generate tweet embeddings for each tweet in this dataset using the roberta-base model
#----------------------------------------------------#
#the following code hasn't been tested
#---------------------------------------------------#
tmp1 <- textEmbed(x     = tweet_data$tweet,
                  model = 'roberta-base',
                  layers = 9:12,
                  context_aggregation_layers = 'concatenate')

tmp1$x

length(tmp3$x)

#append embeddings to original  data


#Task 1.5 Remove the two columns time and tweet from the dataset 

clean_tweet <- tweet_data %>%
  select(-time, -tweet)

#Task 1.6 Prepare a recipe using the recipe() and prep() functions from the recipes 
# package for final transformation of the variables in this dataset.

# 2) List of variable types

outcome <- c('sentiment')

id      <- c('x')

categorical <- c('month') 

numeric   <- c('day',
               'date',
               'hour')

#props      <- c('')

# 3) Convert all nominal, ordinal, and binary variables to factors
# Leave the rest as is

for(i in categorical){
  
  clean_tweet[,i] <- as.factor(clean_tweet[,i])
  
}

# 4) For variables that represent proportions, add/substract a small number
# to 0s/1s for logit transformation

#for(i in props){
#  tweet_data[,i] <- ifelse(tweet_data[,i]==0,.0001,tweet_data[,i])
#  tweet_data[,i] <- ifelse(tweet_data[,i]==1,.9999,tweet_data[,i])
#}

require(recipes)

blueprint <- recipe(x  = clean_tweet,
                    vars  = c(categorical,numeric,outcome,id),
                    roles = c(rep('predictor',4),'outcome','ID')) %>%
  
  # for all 4 predictors, create an indicator variable for missingness
  
  step_indicate_na(all_of(categorical),all_of(numeric)) %>%
  
  # Remove the variable with zero variance, this will also remove the missingness 
  # variables if there is no missingess
  
  step_zv(all_numeric()) %>%
  
  # Impute the missing values using mean and mode. You can instead use a 
  # more advanced imputation model such as bagged trees. I haven't used it due
  # to time concerns
  
  step_impute_mean(all_of(numeric)) %>%
  step_impute_mode(all_of(categorical)) %>%
  
  #Logit transformation of proportions
  
  #step_logit(all_of(props)) %>%
  
  # Natural splines for numeric variables and proportions
  
  step_ns(all_of(numeric),deg_free=3) %>%
  
  # Standardize the natural splines of numeric variables and proportions
  
  step_normalize(paste0(numeric,'_ns_1'),
                 paste0(numeric,'_ns_2'),
                 paste0(numeric,'_ns_3')) %>%
  # paste0(props,'_ns_1'),
  # paste0(props,'_ns_2'),
  # paste0(props,'_ns_3')) %>%
  
  # One-hot encoding for all categorical variables
  
  step_dummy(all_of(categorical),one_hot=TRUE)

blueprint

#Task 1.7
prepare <- prep(blueprint, 
                training = tweet_data)
prepare

baked_tweet_daata <- bake(prepare, new_data = tweet_data)

baked_tweet_data

#Task 1.8 Remove the original day,date, and hour variables from the dataset 

#Task 1.9 Export the final dataset (1500 x 778) as a .csv file and 
# upload it to Canvas along your submission.
