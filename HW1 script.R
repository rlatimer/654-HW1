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

tweet <- tweet_data$tweet

tmp1 <- textEmbed(x     = tweet,
                  model = 'roberta-base',
                  layers = 12,
                  context_aggregation_layers = 'concatenate')

tmp1$x

length(tmp1$x)

#append embeddings to original  data
tweet_emb <- cbind(tweet_data,
                   as.data.frame(tmp1$x)
)

#Task 1.5 Remove the two columns time and tweet from the dataset 

clean_tweet <- tweet_emb %>%
  select(-time, -tweet)

#Task 1.6 Prepare a recipe using the recipe() and prep() functions from the recipes 
# package for final transformation of the variables in this dataset.

#download developer version of recipes
require(devtools)
devtools::install_github("tidymodels/recipes")

#Separate variables

outcome <- c('sentiment')

categorical <- c('month') 

numeric   <- paste0('Dim',1:768)

harmonic <- c('day',
              'date',
              'hour')

#convert variables to factors
for(i in categorical){
  
  clean_tweet[,i] <- as.factor(clean_tweet[,i])
  
}
#Prework on cyclical variables - attempting to create sin and cos columns outside
#  of main data to then add to the overall recipe 
#days sin and cos
#RL note: don't think we need this, I added lines for each harmonic variable below
#day_frame <- data.frame(clean_tweet$day,
#                        x = 1:1500)

#day_frame$x1 <- sin((2*pi*day_frame$x)/7)
#day_frame$x2 <- cos((2*pi*day_frame$x)/7)

#all_tweets <- bind_cols(day_frame, clean_tweet)

#Blueprint development

require(recipes)

blueprint <- recipe(x  = clean_tweet,
                    vars  = c(categorical,numeric,harmonic, outcome),
                    roles = c(rep('predictor',772),'outcome')) %>%
  #cyclical variables
  step_harmonic('day', frequency = 1/7, cycle_size = 1) %>%
  step_harmonic('date', frequency = 1/31, cycle_size = 1) %>%
  step_harmonic('hour', frequency = 1/24, cycle_size = 1) %>%
  #month recorded into dummy variables
  step_dummy(month,one_hot=TRUE)%>%
  #all numeric embeddings are standarized
  # step_ns(all_of(numeric),deg_free=3) %>%
  step_normalize(all_of(numeric))
blueprint

#Task 1.7
prepare <- prep(blueprint, 
                training = clean_tweet)
prepare

baked_tweet_data <- bake(prepare, new_data = clean_tweet)

baked_tweet_data

#Task 1.8 Remove the original day,date, and hour variables from the dataset 

finished_pie <- baked_tweet_data %>%
  select(-day, -date, -hour)

#Task 1.9 Export the final dataset (1500 x 778) as a .csv file and 
# upload it to Canvas along your submission.
#this looks a little different for cloud vs desktop
write.csv(finished_pie,"/home/latimer/hw1/HW1_finished_pie.csv", row.names = FALSE)

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

#table to display frequency of unique categorical variables
cat <- ORtest %>%
  select(2,3,6:26)

lapply(cat, table)

#trgt_assist_fg has N, y, and Y variables
#verifying class
class(ORtest$trgt_assist_fg)
#modified y to Y
ORtest$trgt_assist_fg[ORtest$trgt_assist_fg == "y"] <- "Y"

#confirm change was made:
#table(ORtest$trgt_assist_fg)

#display distribution of numeric variables
numeric <- ORtest %>%
  select(1,4,27:29)

require(ggplot2)
p <- ggplot(numeric, aes(x=enrl_grd, y=score)) + 
  geom_point()
p

#Task 2.5 
#Prepare a recipe
#See homework instructions for specific order of variables
require(recipes)
# List of variable types

outcome <- c('score')

id      <- c('id')

categorical <- c('sex','ethnic_cd','tst_bnch','migrant_ed_fg',
                 'ind_ed_fg','sp_ed_fg','tag_ed_fg','econ_dsvntg','stay_in_dist',
                 'stay_in_schl','dist_sped','trgt_assist_fg',
                 'ayp_dist_partic','ayp_schl_partic','ayp_dist_prfrm',
                 'ayp_schl_prfrm','rc_dist_partic','rc_schl_partic','rc_dist_prfrm',
                 'rc_schl_prfrm','grp_rpt_dist_partic','grp_rpt_schl_partic',
                 'grp_rpt_dist_prfrm','grp_rpt_schl_prfrm') 

numeric   <- c('enrl_grd')

cyclic <- c('date', 'month')
# Convert all nominal, ordinal, and binary variables to factors
# Leave the rest as is

for(i in categorical){
  
  ORtest[,i] <- as.factor(ORtest[,i])
  
}


blueprint2 <- recipe(x  = ORtest,
                    vars  = c(categorical,numeric,cyclic,outcome,id),
                    roles = c(rep('predictor',27),'outcome','id')) %>%
  
  # for all 48 predictors, create an indicator variable for missingness
  
  step_indicate_na(all_of(categorical),all_of(numeric),all_of(cyclic)) %>%
  
  # Remove the variable with zero variance, this will also remove the missingness 
  # variables if there is no missingess
  
  step_zv(all_numeric()) %>%
  
  # Impute the missing values using mean and mode.
  
  step_impute_mean(all_of(numeric)) %>%
  step_impute_mode(all_of(categorical)) %>%
  
  #recode cyclic predictors into two new variables of sin and cos terms,
  step_harmonic('date', frequency = 1/31, cycle_size = 1) %>%
  step_harmonic('month', frequency = 1/12, cycle_size = 1) %>%
  
  # Natural splines for numeric variables and proportions
  
  step_ns(all_of(numeric),deg_free=3) %>%
  
  # Standardize the natural splines of numeric variables
  
  step_normalize(paste0(numeric,'_ns_1'),
                 paste0(numeric,'_ns_2'),
                 paste0(numeric,'_ns_3')) %>%
  
  # One-hot encoding for all categorical variables
  
  step_dummy(all_of(categorical),one_hot=TRUE )

blueprint2

prepare2 <- prep(blueprint2, 
                 training = ORtest)
prepare2

#Task 2.6
#Apply the recipe to the whole dataset

baked_ORtest <- bake(prepare2, new_data = ORtest)

baked_ORtest

#Task 2.7
#Remove the original date and month variables

finished_ORtest <- baked_ORtest %>%
  select(-date, -month)

#Task 2.8
#Export the final dataset (189,426 x 74) as a .csv file
#different for cloud vs desktop
write.csv(finished_pie,"/home/latimer/hw1/HW1_finished_ORtest.csv", row.names = FALSE)



