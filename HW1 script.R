#TASK 1
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
table(tweet_data$day)
table(tweet_data$month)
table(tweet_data$date)
table(tweet_data$hour)


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
tweet <- tweet_data$tweet

tmp1 <- textEmbed(x     = tweet,
                  model = 'roberta-base',
                  layers = 9:12,
                  context_aggregation_layers = 'concatenate')

tmp1$x

length(tmp1$x)

#append embeddings to original  data


#Task 1.5 Remove the two columns time and tweet from the dataset 

clean_tweet <- tweet_data %>%
  select(-time, -tweet)

#Task 1.6 Prepare a recipe using the recipe() and prep() functions from the recipes 
# package for final transformation of the variables in this dataset.

#download developer version of recipes
require(devtools)
devtools::install_github("tidymodels/recipes")

#Separate variables

outcome <- c('sentiment')

id      <- c('x')

categorical <- c('month') 

numeric   <- c('day',
               'date',
               'hour')

#convert variables to factors
for(i in categorical){
  
  clean_tweet[,i] <- as.factor(clean_tweet[,i])
  
}
#Prework on cyclical variables - attempting to create sin and cos columns outside
#  of main data to then add to the overall recipe 
#days sin and cos
day_frame <- data.frame(clean_tweet$day,
                        x = 1:1500)

day_frame$x1 <- sin((2*pi*day_frame$x)/7)
day_frame$x2 <- cos((2*pi*day_frame$x)/7)

all_tweets <- bind_cols(day_frame, clean_tweet)

#Blueprint development

require(recipes)

blueprint <- recipe(x  = clean_tweet,
                    vars  = c(categorical,numeric,outcome,id),
                    roles = c(rep('predictor',4),'outcome','ID')) %>%
#cyclical variables
                    step_harmonic(all_tweets, frequency = 1/7, cycle_size = 1) %>%
#month recorded into dummy variables
                    step_dummy(all_of(categorical),one_hot=TRUE)%>%
#all numeric embeddings are standarized
                    step_ns(all_of(numeric),all_of(props),deg_free=3) %>%
                      step_normalize(paste0(numeric,'_ns_1'),
                                     paste0(numeric,'_ns_2'),
                                     paste0(numeric,'_ns_3'),
                                     paste0(props,'_ns_1'),
                                     paste0(props,'_ns_2'),
                                     paste0(props,'_ns_3'))

blueprint

#######################################################################################
#from lecture notes

# 2) List of variable types

#outcome <- c('sentiment')

#id      <- c('x')

#categorical <- c('month') 

#numeric   <- c('day',
#               'date',
#               'hour')

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

finished_pie <- baked_tweet_data %>%
  select(-day, -date, -hour)

#Task 1.9 Export the final dataset (1500 x 778) as a .csv file and 
# upload it to Canvas along your submission.


#TASK 2
#Task 2.1
#Import Oregon Testing Data
ortest_data <- read.csv('https://raw.githubusercontent.com/uo-datasci-specialization/c4-ml-fall-2021/main/data/oregon.csv')

#Task 2.2
#Create two new columns to show the date and month the test was taken (as numeric variables)

#separate tst_dt column into a vector
x <- ortest_data$tst_dt
strsplit(x,'/')

#separate month from vector
sapply(strsplit(x,'/'),`[`,1)

#make month numeric
ortest_data$month <- as.numeric(sapply(strsplit(x,'/'),`[`,1))

#separate day and make numeric
ortest_data$date <- as.numeric(sapply(strsplit(x,'/'),`[`,2))

#Remove the column tst_dt from the dataset 
clean_ortest <- ortest_data %>%
  select(-tst_dt)

#Calculate and print the frequencies for the new columns (date and month)
table(clean_ortest$month)
table(clean_ortest$date)

#Task 2.3
#Use the ff_glimpse() function from the finalfit package to provide a snapshot of missingness 
#this function also returns the number of levels for categorical variables
#Remove any variable with large amount of missingness (e.g. more than 75%)

#install.packages("finalfit")
library(finalfit)

dependent = NULL
explanatory = NULL
clean_ortest %>%
  ff_glimpse(dependent, explanatory)

#removing variables above 75%
ORtest <- clean_ortest %>%
  select(-ayp_lep)

#Task 2.4
#Most of the variables in this dataset are categorical (binary variable with a Yes and No response)
#Check the frequency of unique values for all categorical variables
#If there is any inconsistency (e.g., Yes is coded as both 'y' and 'Y') for any of these variables 
#in terms of how values are coded, fix them. 
#Also, check the distribution of numeric variables and make sure there is no anomaly.


#Task 2.5 
#Prepare a recipe
#See homework instructions for specific order of variables

#Task 2.6
#Apply the recipe to the whole dataset

#Task 2.7
#Remove the original date and month variables

#Task 2.8
#Export the final dataset (189,426 x 74) as a .csv file




