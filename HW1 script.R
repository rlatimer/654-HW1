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

numeric   <- c(
  'Dim1','Dim2','Dim3','Dim4','Dim5', 'Dim6', 'Dim7', 
  'Dim8', 'Dim9','Dim10','Dim11','Dim12','Dim13',
  'Dim14', 'Dim15', 'Dim16', 'Dim17', 'Dim18', 'Dim19',
  'Dim20', 'Dim21', 'Dim22', 'Dim23', 'Dim24', 'Dim25',
  'Dim26', 'Dim27', 'Dim28', 'Dim29', 'Dim30', 'Dim31',
  'Dim32', 'Dim33', 'Dim34', 'Dim35', 'Dim36', 'Dim37', 
  'Dim38', 'Dim39', 'Dim40', 'Dim41', 'Dim42', 'Dim43',
  'Dim44', 'Dim45', 'Dim46', 'Dim47', 'Dim48', 'Dim49', 
  'Dim50', 'Dim51', 'Dim52', 'Dim53', 'Dim54', 'Dim55', 
  'Dim56', 'Dim57', 'Dim58', 'Dim59', 'Dim60', 'Dim61', 
  'Dim62', 'Dim63', 'Dim64', 'Dim65', 'Dim66', 'Dim67', 
  'Dim68', 'Dim69', 'Dim70', 'Dim71', 'Dim72', 'Dim73', 
  'Dim74', 'Dim75', 'Dim76', 'Dim77', 'Dim78', 'Dim79', 
  'Dim80', 'Dim81', 'Dim82', 'Dim83', 'Dim84', 'Dim85', 
  'Dim86', 'Dim87', 'Dim88', 'Dim89', 'Dim90', 'Dim91', 
  'Dim92', 'Dim93', 'Dim94', 'Dim95', 'Dim96', 'Dim97', 
  'Dim98', 'Dim99', 'Dim100', 'Dim101', 'Dim102', 'Dim103', 
  'Dim104', 'Dim105', 'Dim106', 'Dim107', 'Dim108', 'Dim109', 
  'Dim110', 'Dim111', 'Dim112', 'Dim113', 'Dim114', 'Dim115', 
  'Dim116', 'Dim117', 'Dim118', 'Dim119', 'Dim120', 'Dim121', 
  'Dim122', 'Dim123', 'Dim124', 'Dim125', 'Dim126', 'Dim127', 
  'Dim128', 'Dim129', 'Dim130', 'Dim131', 'Dim132', 'Dim133', 
  'Dim134', 'Dim135', 'Dim136', 'Dim137', 'Dim138', 'Dim139', 
  'Dim140', 'Dim141', 'Dim142', 'Dim143', 'Dim144', 'Dim145', 
  'Dim146', 'Dim147', 'Dim148', 'Dim149', 'Dim150', 'Dim151', 
  'Dim152', 'Dim153', 'Dim154', 'Dim155', 'Dim156', 'Dim157', 
  'Dim158', 'Dim159', 'Dim160', 'Dim161', 'Dim162', 'Dim163', 
  'Dim164', 'Dim165', 'Dim166', 'Dim167', 'Dim168', 'Dim169', 
  'Dim170', 'Dim171', 'Dim172', 'Dim173', 'Dim174', 'Dim175', 
  'Dim176', 'Dim177', 'Dim178', 'Dim179', 'Dim180', 'Dim181', 
  'Dim182', 'Dim183', 'Dim184', 'Dim185', 'Dim186', 'Dim187', 
  'Dim188', 'Dim189', 'Dim190', 'Dim191', 'Dim192', 'Dim193', 
  'Dim194', 'Dim195', 'Dim196', 'Dim197', 'Dim198', 'Dim199', 
  'Dim200', 'Dim201', 'Dim202', 'Dim203', 'Dim204', 'Dim205', 
  'Dim206', 'Dim207', 'Dim208', 'Dim209', 'Dim210', 'Dim211', 
  'Dim212', 'Dim213', 'Dim214', 'Dim215', 'Dim216', 'Dim217', 
  'Dim218', 'Dim219', 'Dim220', 'Dim221', 'Dim222', 'Dim223', 
  'Dim224', 'Dim225', 'Dim226', 'Dim227', 'Dim228', 'Dim229', 
  'Dim230', 'Dim231', 'Dim232', 'Dim233', 'Dim234', 'Dim235', 
  'Dim236', 'Dim237', 'Dim238', 'Dim239', 'Dim240', 'Dim241', 
  'Dim242', 'Dim243', 'Dim244', 'Dim245', 'Dim246', 'Dim247', 
  'Dim248', 'Dim249', 'Dim250', 'Dim251', 'Dim252', 'Dim253', 
  'Dim254', 'Dim255', 'Dim256', 'Dim257', 'Dim258', 'Dim259', 
  'Dim260', 'Dim261', 'Dim262', 'Dim263', 'Dim264', 'Dim265', 
  'Dim266', 'Dim267', 'Dim268', 'Dim269', 'Dim270', 'Dim271', 
  'Dim272', 'Dim273', 'Dim274', 'Dim275', 'Dim276', 'Dim277', 
  'Dim278', 'Dim279', 'Dim280', 'Dim281', 'Dim282', 'Dim283', 
  'Dim284', 'Dim285', 'Dim286', 'Dim287', 'Dim288', 'Dim289', 
  'Dim290', 'Dim291', 'Dim292', 'Dim293', 'Dim294', 'Dim295', 
  'Dim296', 'Dim297', 'Dim298', 'Dim299', 'Dim300', 'Dim301', 
  'Dim302', 'Dim303', 'Dim304', 'Dim305', 'Dim306', 'Dim307', 
  'Dim308', 'Dim309', 'Dim310', 'Dim311', 'Dim312', 'Dim313', 
  'Dim314', 'Dim315', 'Dim316', 'Dim317', 'Dim318', 'Dim319', 
  'Dim320', 'Dim321', 'Dim322', 'Dim323', 'Dim324', 'Dim325', 
  'Dim326', 'Dim327', 'Dim328', 'Dim329', 'Dim330', 'Dim331', 
  'Dim332', 'Dim333', 'Dim334', 'Dim335', 'Dim336', 'Dim337', 
  'Dim338', 'Dim339', 'Dim340', 'Dim341', 'Dim342', 'Dim343', 
  'Dim344', 'Dim345', 'Dim346', 'Dim347', 'Dim348', 'Dim349', 
  'Dim350', 'Dim351', 'Dim352', 'Dim353', 'Dim354', 'Dim355', 
  'Dim356', 'Dim357', 'Dim358', 'Dim359', 'Dim360', 'Dim361', 
  'Dim362', 'Dim363', 'Dim364', 'Dim365', 'Dim366', 'Dim367', 
  'Dim368', 'Dim369', 'Dim370', 'Dim371', 'Dim372', 'Dim373', 
  'Dim374', 'Dim375', 'Dim376', 'Dim377', 'Dim378', 'Dim379', 
  'Dim380', 'Dim381', 'Dim382', 'Dim383', 'Dim384', 'Dim385', 
  'Dim386', 'Dim387', 'Dim388', 'Dim389', 'Dim390', 'Dim391', 
  'Dim392', 'Dim393', 'Dim394', 'Dim395', 'Dim396', 'Dim397', 
  'Dim398', 'Dim399', 'Dim400', 'Dim401', 'Dim402', 'Dim403', 
  'Dim404', 'Dim405', 'Dim406', 'Dim407', 'Dim408', 'Dim409', 
  'Dim410', 'Dim411', 'Dim412', 'Dim413', 'Dim414', 'Dim415', 
  'Dim416', 'Dim417', 'Dim418', 'Dim419', 'Dim420', 'Dim421', 
  'Dim422', 'Dim423', 'Dim424', 'Dim425', 'Dim426', 'Dim427', 
  'Dim428', 'Dim429', 'Dim430', 'Dim431', 'Dim432', 'Dim433', 
  'Dim434', 'Dim435', 'Dim436', 'Dim437', 'Dim438', 'Dim439', 
  'Dim440', 'Dim441', 'Dim442', 'Dim443', 'Dim444', 'Dim445', 
  'Dim446', 'Dim447', 'Dim448', 'Dim449', 'Dim450', 'Dim451', 
  'Dim452', 'Dim453', 'Dim454', 'Dim455', 'Dim456', 'Dim457', 
  'Dim458', 'Dim459', 'Dim460', 'Dim461', 'Dim462', 'Dim463', 
  'Dim464', 'Dim465', 'Dim466', 'Dim467', 'Dim468', 'Dim469', 
  'Dim470', 'Dim471', 'Dim472', 'Dim473', 'Dim474', 'Dim475', 
  'Dim476', 'Dim477', 'Dim478', 'Dim479', 'Dim480', 'Dim481', 
  'Dim482', 'Dim483', 'Dim484', 'Dim485', 'Dim486', 'Dim487', 
  'Dim488', 'Dim489', 'Dim490', 'Dim491', 'Dim492', 'Dim493', 
  'Dim494', 'Dim495', 'Dim496', 'Dim497', 'Dim498', 'Dim499', 
  'Dim500', 'Dim501', 'Dim502', 'Dim503', 'Dim504', 'Dim505', 
  'Dim506', 'Dim507', 'Dim508', 'Dim509', 'Dim510', 'Dim511', 
  'Dim512', 'Dim513', 'Dim514', 'Dim515', 'Dim516', 'Dim517', 
  'Dim518', 'Dim519', 'Dim520', 'Dim521', 'Dim522', 'Dim523', 
  'Dim524', 'Dim525', 'Dim526', 'Dim527', 'Dim528', 'Dim529', 
  'Dim530', 'Dim531', 'Dim532', 'Dim533', 'Dim534', 'Dim535', 
  'Dim536', 'Dim537', 'Dim538', 'Dim539', 'Dim540', 'Dim541', 
  'Dim542', 'Dim543', 'Dim544', 'Dim545', 'Dim546', 'Dim547', 
  'Dim548', 'Dim549', 'Dim550', 'Dim551', 'Dim552', 'Dim553', 
  'Dim554', 'Dim555', 'Dim556', 'Dim557', 'Dim558', 'Dim559', 
  'Dim560', 'Dim561', 'Dim562', 'Dim563', 'Dim564', 'Dim565', 
  'Dim566', 'Dim567', 'Dim568', 'Dim569', 'Dim570', 'Dim571', 
  'Dim572', 'Dim573', 'Dim574', 'Dim575', 'Dim576', 'Dim577', 
  'Dim578', 'Dim579', 'Dim580', 'Dim581', 'Dim582', 'Dim583', 
  'Dim584', 'Dim585', 'Dim586', 'Dim587', 'Dim588', 'Dim589', 
  'Dim590', 'Dim591', 'Dim592', 'Dim593', 'Dim594', 'Dim595', 
  'Dim596', 'Dim597', 'Dim598', 'Dim599', 'Dim600', 'Dim601', 
  'Dim602', 'Dim603', 'Dim604', 'Dim605', 'Dim606', 'Dim607', 
  'Dim608', 'Dim609', 'Dim610', 'Dim611', 'Dim612', 'Dim613', 
  'Dim614', 'Dim615', 'Dim616', 'Dim617', 'Dim618', 'Dim619', 
  'Dim620', 'Dim621', 'Dim622', 'Dim623', 'Dim624', 'Dim625', 
  'Dim626', 'Dim627', 'Dim628', 'Dim629', 'Dim630', 'Dim631', 
  'Dim632', 'Dim633', 'Dim634', 'Dim635', 'Dim636', 'Dim637', 
  'Dim638', 'Dim639', 'Dim640', 'Dim641', 'Dim642', 'Dim643', 
  'Dim644', 'Dim645', 'Dim646', 'Dim647', 'Dim648', 'Dim649', 
  'Dim650', 'Dim651', 'Dim652', 'Dim653', 'Dim654', 'Dim655', 
  'Dim656', 'Dim657', 'Dim658', 'Dim659', 'Dim660', 'Dim661', 
  'Dim662', 'Dim663', 'Dim664', 'Dim665', 'Dim666', 'Dim667', 
  'Dim668', 'Dim669', 'Dim670', 'Dim671', 'Dim672', 'Dim673', 
  'Dim674', 'Dim675', 'Dim676', 'Dim677', 'Dim678', 'Dim679', 
  'Dim680', 'Dim681', 'Dim682', 'Dim683', 'Dim684', 'Dim685',
  'Dim686', 'Dim687', 'Dim688', 'Dim689', 'Dim690', 'Dim691', 
  'Dim692', 'Dim693', 'Dim694', 'Dim695', 'Dim696', 'Dim697', 
  'Dim698', 'Dim699', 'Dim700', 'Dim701', 'Dim702', 'Dim703', 
  'Dim704', 'Dim705', 'Dim706', 'Dim707', 'Dim708', 'Dim709', 
  'Dim710', 'Dim711', 'Dim712', 'Dim713', 'Dim714', 'Dim715', 
  'Dim716', 'Dim717', 'Dim718', 'Dim719', 'Dim720', 'Dim721', 
  'Dim722', 'Dim723', 'Dim724', 'Dim725', 'Dim726', 'Dim727',
  'Dim728', 'Dim729', 'Dim730', 'Dim731', 'Dim732', 'Dim733', 
  'Dim734', 'Dim735', 'Dim736', 'Dim737', 'Dim738', 'Dim739', 
  'Dim740', 'Dim741', 'Dim742', 'Dim743', 'Dim744', 'Dim745', 
  'Dim746', 'Dim747', 'Dim748', 'Dim749', 'Dim750', 'Dim751', 
  'Dim752', 'Dim753', 'Dim754', 'Dim755', 'Dim756', 'Dim757', 
  'Dim758', 'Dim759', 'Dim760', 'Dim761', 'Dim762', 'Dim763', 
  'Dim764', 'Dim765', 'Dim766', 'Dim767', 'Dim768')

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

require(recipes)

blueprint <- recipe(x  = clean_tweet,
                    vars  = c(categorical,numeric,harmonic, outcome),
                    roles = c(rep('predictor',772),'outcome')) %>%
  #cyclical variables
  step_harmonic(day, frequency = 1/7, cycle_size = 1) %>%
  step_harmonic(date, frequency = 1/31, cycle_size = 1) %>%
  step_harmonic(hour, frequency = 1/24, cycle_size = 1) %>%
  #month recorded into dummy variables
  step_dummy(month,one_hot=TRUE)%>%
  #all numeric embeddings are standarized
  # step_ns(all_of(numeric),deg_free=3) %>%
  step_normalize(
    'Dim1','Dim2','Dim3','Dim4','Dim5', 'Dim6', 'Dim7', 
    'Dim8', 'Dim9','Dim10','Dim11','Dim12','Dim13',
    'Dim14', 'Dim15', 'Dim16', 'Dim17', 'Dim18', 'Dim19',
    'Dim20', 'Dim21', 'Dim22', 'Dim23', 'Dim24', 'Dim25',
    'Dim26', 'Dim27', 'Dim28', 'Dim29', 'Dim30', 'Dim31',
    'Dim32', 'Dim33', 'Dim34', 'Dim35', 'Dim36', 'Dim37', 
    'Dim38', 'Dim39', 'Dim40', 'Dim41', 'Dim42', 'Dim43',
    'Dim44', 'Dim45', 'Dim46', 'Dim47', 'Dim48', 'Dim49', 
    'Dim50', 'Dim51', 'Dim52', 'Dim53', 'Dim54', 'Dim55', 
    'Dim56', 'Dim57', 'Dim58', 'Dim59', 'Dim60', 'Dim61', 
    'Dim62', 'Dim63', 'Dim64', 'Dim65', 'Dim66', 'Dim67', 
    'Dim68', 'Dim69', 'Dim70', 'Dim71', 'Dim72', 'Dim73', 
    'Dim74', 'Dim75', 'Dim76', 'Dim77', 'Dim78', 'Dim79', 
    'Dim80', 'Dim81', 'Dim82', 'Dim83', 'Dim84', 'Dim85', 
    'Dim86', 'Dim87', 'Dim88', 'Dim89', 'Dim90', 'Dim91', 
    'Dim92', 'Dim93', 'Dim94', 'Dim95', 'Dim96', 'Dim97', 
    'Dim98', 'Dim99', 'Dim100', 'Dim101', 'Dim102', 'Dim103', 
    'Dim104', 'Dim105', 'Dim106', 'Dim107', 'Dim108', 'Dim109', 
    'Dim110', 'Dim111', 'Dim112', 'Dim113', 'Dim114', 'Dim115', 
    'Dim116', 'Dim117', 'Dim118', 'Dim119', 'Dim120', 'Dim121', 
    'Dim122', 'Dim123', 'Dim124', 'Dim125', 'Dim126', 'Dim127', 
    'Dim128', 'Dim129', 'Dim130', 'Dim131', 'Dim132', 'Dim133', 
    'Dim134', 'Dim135', 'Dim136', 'Dim137', 'Dim138', 'Dim139', 
    'Dim140', 'Dim141', 'Dim142', 'Dim143', 'Dim144', 'Dim145', 
    'Dim146', 'Dim147', 'Dim148', 'Dim149', 'Dim150', 'Dim151', 
    'Dim152', 'Dim153', 'Dim154', 'Dim155', 'Dim156', 'Dim157', 
    'Dim158', 'Dim159', 'Dim160', 'Dim161', 'Dim162', 'Dim163', 
    'Dim164', 'Dim165', 'Dim166', 'Dim167', 'Dim168', 'Dim169', 
    'Dim170', 'Dim171', 'Dim172', 'Dim173', 'Dim174', 'Dim175', 
    'Dim176', 'Dim177', 'Dim178', 'Dim179', 'Dim180', 'Dim181', 
    'Dim182', 'Dim183', 'Dim184', 'Dim185', 'Dim186', 'Dim187', 
    'Dim188', 'Dim189', 'Dim190', 'Dim191', 'Dim192', 'Dim193', 
    'Dim194', 'Dim195', 'Dim196', 'Dim197', 'Dim198', 'Dim199', 
    'Dim200', 'Dim201', 'Dim202', 'Dim203', 'Dim204', 'Dim205', 
    'Dim206', 'Dim207', 'Dim208', 'Dim209', 'Dim210', 'Dim211', 
    'Dim212', 'Dim213', 'Dim214', 'Dim215', 'Dim216', 'Dim217', 
    'Dim218', 'Dim219', 'Dim220', 'Dim221', 'Dim222', 'Dim223', 
    'Dim224', 'Dim225', 'Dim226', 'Dim227', 'Dim228', 'Dim229', 
    'Dim230', 'Dim231', 'Dim232', 'Dim233', 'Dim234', 'Dim235', 
    'Dim236', 'Dim237', 'Dim238', 'Dim239', 'Dim240', 'Dim241', 
    'Dim242', 'Dim243', 'Dim244', 'Dim245', 'Dim246', 'Dim247', 
    'Dim248', 'Dim249', 'Dim250', 'Dim251', 'Dim252', 'Dim253', 
    'Dim254', 'Dim255', 'Dim256', 'Dim257', 'Dim258', 'Dim259', 
    'Dim260', 'Dim261', 'Dim262', 'Dim263', 'Dim264', 'Dim265', 
    'Dim266', 'Dim267', 'Dim268', 'Dim269', 'Dim270', 'Dim271', 
    'Dim272', 'Dim273', 'Dim274', 'Dim275', 'Dim276', 'Dim277', 
    'Dim278', 'Dim279', 'Dim280', 'Dim281', 'Dim282', 'Dim283', 
    'Dim284', 'Dim285', 'Dim286', 'Dim287', 'Dim288', 'Dim289', 
    'Dim290', 'Dim291', 'Dim292', 'Dim293', 'Dim294', 'Dim295', 
    'Dim296', 'Dim297', 'Dim298', 'Dim299', 'Dim300', 'Dim301', 
    'Dim302', 'Dim303', 'Dim304', 'Dim305', 'Dim306', 'Dim307', 
    'Dim308', 'Dim309', 'Dim310', 'Dim311', 'Dim312', 'Dim313', 
    'Dim314', 'Dim315', 'Dim316', 'Dim317', 'Dim318', 'Dim319', 
    'Dim320', 'Dim321', 'Dim322', 'Dim323', 'Dim324', 'Dim325', 
    'Dim326', 'Dim327', 'Dim328', 'Dim329', 'Dim330', 'Dim331', 
    'Dim332', 'Dim333', 'Dim334', 'Dim335', 'Dim336', 'Dim337', 
    'Dim338', 'Dim339', 'Dim340', 'Dim341', 'Dim342', 'Dim343', 
    'Dim344', 'Dim345', 'Dim346', 'Dim347', 'Dim348', 'Dim349', 
    'Dim350', 'Dim351', 'Dim352', 'Dim353', 'Dim354', 'Dim355', 
    'Dim356', 'Dim357', 'Dim358', 'Dim359', 'Dim360', 'Dim361', 
    'Dim362', 'Dim363', 'Dim364', 'Dim365', 'Dim366', 'Dim367', 
    'Dim368', 'Dim369', 'Dim370', 'Dim371', 'Dim372', 'Dim373', 
    'Dim374', 'Dim375', 'Dim376', 'Dim377', 'Dim378', 'Dim379', 
    'Dim380', 'Dim381', 'Dim382', 'Dim383', 'Dim384', 'Dim385', 
    'Dim386', 'Dim387', 'Dim388', 'Dim389', 'Dim390', 'Dim391', 
    'Dim392', 'Dim393', 'Dim394', 'Dim395', 'Dim396', 'Dim397', 
    'Dim398', 'Dim399', 'Dim400', 'Dim401', 'Dim402', 'Dim403', 
    'Dim404', 'Dim405', 'Dim406', 'Dim407', 'Dim408', 'Dim409', 
    'Dim410', 'Dim411', 'Dim412', 'Dim413', 'Dim414', 'Dim415', 
    'Dim416', 'Dim417', 'Dim418', 'Dim419', 'Dim420', 'Dim421', 
    'Dim422', 'Dim423', 'Dim424', 'Dim425', 'Dim426', 'Dim427', 
    'Dim428', 'Dim429', 'Dim430', 'Dim431', 'Dim432', 'Dim433', 
    'Dim434', 'Dim435', 'Dim436', 'Dim437', 'Dim438', 'Dim439', 
    'Dim440', 'Dim441', 'Dim442', 'Dim443', 'Dim444', 'Dim445', 
    'Dim446', 'Dim447', 'Dim448', 'Dim449', 'Dim450', 'Dim451', 
    'Dim452', 'Dim453', 'Dim454', 'Dim455', 'Dim456', 'Dim457', 
    'Dim458', 'Dim459', 'Dim460', 'Dim461', 'Dim462', 'Dim463', 
    'Dim464', 'Dim465', 'Dim466', 'Dim467', 'Dim468', 'Dim469', 
    'Dim470', 'Dim471', 'Dim472', 'Dim473', 'Dim474', 'Dim475', 
    'Dim476', 'Dim477', 'Dim478', 'Dim479', 'Dim480', 'Dim481', 
    'Dim482', 'Dim483', 'Dim484', 'Dim485', 'Dim486', 'Dim487', 
    'Dim488', 'Dim489', 'Dim490', 'Dim491', 'Dim492', 'Dim493', 
    'Dim494', 'Dim495', 'Dim496', 'Dim497', 'Dim498', 'Dim499', 
    'Dim500', 'Dim501', 'Dim502', 'Dim503', 'Dim504', 'Dim505', 
    'Dim506', 'Dim507', 'Dim508', 'Dim509', 'Dim510', 'Dim511', 
    'Dim512', 'Dim513', 'Dim514', 'Dim515', 'Dim516', 'Dim517', 
    'Dim518', 'Dim519', 'Dim520', 'Dim521', 'Dim522', 'Dim523', 
    'Dim524', 'Dim525', 'Dim526', 'Dim527', 'Dim528', 'Dim529', 
    'Dim530', 'Dim531', 'Dim532', 'Dim533', 'Dim534', 'Dim535', 
    'Dim536', 'Dim537', 'Dim538', 'Dim539', 'Dim540', 'Dim541', 
    'Dim542', 'Dim543', 'Dim544', 'Dim545', 'Dim546', 'Dim547', 
    'Dim548', 'Dim549', 'Dim550', 'Dim551', 'Dim552', 'Dim553', 
    'Dim554', 'Dim555', 'Dim556', 'Dim557', 'Dim558', 'Dim559', 
    'Dim560', 'Dim561', 'Dim562', 'Dim563', 'Dim564', 'Dim565', 
    'Dim566', 'Dim567', 'Dim568', 'Dim569', 'Dim570', 'Dim571', 
    'Dim572', 'Dim573', 'Dim574', 'Dim575', 'Dim576', 'Dim577', 
    'Dim578', 'Dim579', 'Dim580', 'Dim581', 'Dim582', 'Dim583', 
    'Dim584', 'Dim585', 'Dim586', 'Dim587', 'Dim588', 'Dim589', 
    'Dim590', 'Dim591', 'Dim592', 'Dim593', 'Dim594', 'Dim595', 
    'Dim596', 'Dim597', 'Dim598', 'Dim599', 'Dim600', 'Dim601', 
    'Dim602', 'Dim603', 'Dim604', 'Dim605', 'Dim606', 'Dim607', 
    'Dim608', 'Dim609', 'Dim610', 'Dim611', 'Dim612', 'Dim613', 
    'Dim614', 'Dim615', 'Dim616', 'Dim617', 'Dim618', 'Dim619', 
    'Dim620', 'Dim621', 'Dim622', 'Dim623', 'Dim624', 'Dim625', 
    'Dim626', 'Dim627', 'Dim628', 'Dim629', 'Dim630', 'Dim631', 
    'Dim632', 'Dim633', 'Dim634', 'Dim635', 'Dim636', 'Dim637', 
    'Dim638', 'Dim639', 'Dim640', 'Dim641', 'Dim642', 'Dim643', 
    'Dim644', 'Dim645', 'Dim646', 'Dim647', 'Dim648', 'Dim649', 
    'Dim650', 'Dim651', 'Dim652', 'Dim653', 'Dim654', 'Dim655', 
    'Dim656', 'Dim657', 'Dim658', 'Dim659', 'Dim660', 'Dim661', 
    'Dim662', 'Dim663', 'Dim664', 'Dim665', 'Dim666', 'Dim667', 
    'Dim668', 'Dim669', 'Dim670', 'Dim671', 'Dim672', 'Dim673', 
    'Dim674', 'Dim675', 'Dim676', 'Dim677', 'Dim678', 'Dim679', 
    'Dim680', 'Dim681', 'Dim682', 'Dim683', 'Dim684', 'Dim685',
    'Dim686', 'Dim687', 'Dim688', 'Dim689', 'Dim690', 'Dim691', 
    'Dim692', 'Dim693', 'Dim694', 'Dim695', 'Dim696', 'Dim697', 
    'Dim698', 'Dim699', 'Dim700', 'Dim701', 'Dim702', 'Dim703', 
    'Dim704', 'Dim705', 'Dim706', 'Dim707', 'Dim708', 'Dim709', 
    'Dim710', 'Dim711', 'Dim712', 'Dim713', 'Dim714', 'Dim715', 
    'Dim716', 'Dim717', 'Dim718', 'Dim719', 'Dim720', 'Dim721', 
    'Dim722', 'Dim723', 'Dim724', 'Dim725', 'Dim726', 'Dim727',
    'Dim728', 'Dim729', 'Dim730', 'Dim731', 'Dim732', 'Dim733', 
    'Dim734', 'Dim735', 'Dim736', 'Dim737', 'Dim738', 'Dim739', 
    'Dim740', 'Dim741', 'Dim742', 'Dim743', 'Dim744', 'Dim745', 
    'Dim746', 'Dim747', 'Dim748', 'Dim749', 'Dim750', 'Dim751', 
    'Dim752', 'Dim753', 'Dim754', 'Dim755', 'Dim756', 'Dim757', 
    'Dim758', 'Dim759', 'Dim760', 'Dim761', 'Dim762', 'Dim763', 
    'Dim764', 'Dim765', 'Dim766', 'Dim767', 'Dim768')
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
categorical <- ORtest %>%
  select(2,3,6:26)

lapply(categorical, table)

#trgt_assist_fg has N, y, and Y variables
#verifying class
class(ORtest$trgt_assist_fg)
#modified y to Y
ORtest$trgt_assist_fg[ORtest$trgt_assist_fg == "y"] <- "Y"

#display distribution of numeric variables


#Task 2.5 
#Prepare a recipe
#See homework instructions for specific order of variables

#Task 2.6
#Apply the recipe to the whole dataset

#Task 2.7
#Remove the original date and month variables

#Task 2.8
#Export the final dataset (189,426 x 74) as a .csv file




