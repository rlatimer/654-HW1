
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

