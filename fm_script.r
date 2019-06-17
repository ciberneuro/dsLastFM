###################################
# Installing and loading libraries
###################################

# Note: Make sure you have properly installed and loaded the libraries before continuing

if(!require(stringi)) install.packages("stringi")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org", dependencies=TRUE)
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org", dependencies=TRUE)
if(!require(data.table)) install.packages("data.table")
if(!require(recommenderlab)) install.packages("recommenderlab")

###################################
# Data loading and exploration
###################################

dl1 <- tempfile()
download.file("https://raw.githubusercontent.com/ciberneuro/dsLastFM/master/data/user_artists.dat", dl1)
plays <- fread(text = readLines(dl1), header = TRUE)
dl2 <- tempfile()
download.file("https://raw.githubusercontent.com/ciberneuro/dsLastFM/master/data/artists.dat", dl2)
artists <- fread(text = readLines(dl2), header = TRUE,
                 quote="", drop=c("url", "pictureURL"),
                 col.names = c("artistID", "artistName"))
rm(dl1, dl2)
artists$artistName <- stri_trans_general(artists$artistName, "Any-Latin")
print(paste("The dataset has", dim(plays)[1], "rows and", dim(plays)[2], "columns."))
print(paste("The columns names are: ", toString(colnames(plays))))

#Detecting outliers in plays dataset
hist(plays$weight, main="Histogram of number of plays")
writeLines(paste("The plays have an average of", mean(plays$weight),
                 "\nwith a median of", median(plays$weight),
                 "\nand a maximum of", max(plays$weight)))
outlier_limit <- boxplot.stats(plays$weight)$stats[5]
writeLines(paste("The statistical process that R uses for plotting,",
                 "\ndefines the outliers as the numbers above", outlier_limit))

#Bounding the data above 1000
plays$weight[plays$weight>1000] <- 1000

#Detecting many 1 user artists
artist_plays <- plays %>% 
  group_by(artistID) %>% 
  summarize(n=n(), weight = mean(weight)) %>%
  left_join(artists, by="artistID")
artist_plays %>% arrange(desc(weight)) %>% top_n(10,weight)

#Removing artists with less than 25 users
plays <- plays %>% 
  semi_join(artist_plays %>% filter(n>=25), by="artistID")

#More popular artists in the top table
artist_plays <- plays %>% 
  group_by(artistID) %>% 
  summarize(n=n(), weight = mean(weight)) %>%
  left_join(artists, by="artistID")
artist_plays %>% arrange(desc(weight)) %>% top_n(10,weight)

#Checking filtered and capped dataset
print(paste("The dataset has", dim(plays)[1], "rows and", dim(plays)[2], "columns."))
print(paste("The columns names are: ", toString(colnames(plays))))

#About the users
print(paste("There are", length(unique(plays$userID)), "unique users"))
user_plays <- plays %>% group_by(userID) %>% summarize(n=n(), w = sum(weight))
hist(user_plays$n, main ="Histogram of number of artists played per user")
writeLines(paste("The average number of artists played by a user is",
                 mean(user_plays$n)))
hist(user_plays$w , main="Histogram of number of plays per user")
writeLines(paste("Users have played an average of", mean(user_plays$w), "songs,",
                 "\nwith a median of", median(user_plays$w),
                 "\nand a maximum of", max(user_plays$w)
))

#About the artists
print(paste("There are", length(unique(plays$artistID)), "unique artists."))
writeLines(paste("The average artist is listened ", mean(artist_plays$weight),
                 "times\nby", mean(artist_plays$n), "users." ))


#Applying Penalized Least Square method
#fist we generate train and validation sets
set.seed(1)
test_index <- createDataPartition(y = plays$weight, times = 1, p = 0.1, list = FALSE)
trainx <- plays[-test_index,]
temp <- plays[test_index,]

validation <- temp %>% 
  semi_join(trainx, by = "artistID") %>%
  semi_join(trainx, by = "userID")

removed <- anti_join(temp, validation)
trainx <- rbind(trainx, removed)
rm(test_index, temp, removed)

#Adding user that likes Bloc Party, Incubus, Red Hot Chili Peppers and Arctic Monkeys
trainx <- rbind(trainx, data.frame(
  userID = c(9999, 9999, 9999, 9999),
  artistID = c(210, 1116, 220, 207),
  weight = c(1000, 1000, 1000, 1000)
))

#optimizing Lambda
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(trainx$weight)
  
  b_i <- trainx %>% 
    group_by(artistID) %>%
    summarize(b_i = sum(weight - mu)/(n()+l))
  
  b_u <- trainx %>% 
    left_join(b_i, by="artistID") %>%
    group_by(userID) %>%
    summarize(b_u = sum(weight - b_i - mu)/(n()+l))
  
  predicted_values <- 
    trainx %>% 
    left_join(b_i, by = "artistID") %>%
    left_join(b_u, by = "userID") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_values, trainx$weight))
})
qplot(lambdas, rmses)
print(paste("Lambda", lambdas[which.min(rmses)], "offers the lowest RMSE."))

#Testing Penalized Least Square method
lambda <- 0.25
mu <- mean(trainx$weight)

b_i <- trainx %>% 
  group_by(artistID) %>%
  summarize(b_i = sum(weight - mu)/(n()+lambda))

b_u <- trainx %>% 
  left_join(b_i, by="artistID") %>%
  group_by(userID) %>%
  summarize(b_u = sum(weight - b_i - mu)/(n()+lambda))

predicted_values <- 
  validation %>% 
  left_join(b_i, by = "artistID") %>%
  left_join(b_u, by = "userID") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

validation_RMSE <- RMSE(predicted_values, validation$weight)
print(paste("Our algorithm has a RMSE of", validation_RMSE))

#Penalized Least Square suggestions
suggestions <- merge(data.frame(userID = 9999), artists, all=TRUE) %>%
  left_join(b_i, by = "artistID") %>%
  left_join(b_u, by = "userID") %>%
  semi_join(trainx, by = "artistID") %>%
  mutate(pred = mu + b_i + b_u, artistName = stringr::str_trunc(artistName, 40))
suggestions %>%
  select(artistID, artistName, pred) %>%
  arrange(desc(pred)) %>% top_n(10, pred)

#Suggestions with a different b_u
suggestions <- merge(data.frame(userID = 9999), artists, all=TRUE) %>%
  left_join(b_i, by = "artistID") %>%
  semi_join(trainx, by = "artistID") %>%
  mutate(pred = mu + b_i + 1, artistName = stringr::str_trunc(artistName, 40))
suggestions %>%
  select(artistID, artistName, pred) %>%
  arrange(desc(pred)) %>% top_n(10, pred)

#adding two users with different tastes to plays dataset
#Note: try changing the artistID in user 9998
plays <- rbind(plays, data.frame(
  userID = c(9998, 9998, 9998),
  artistID = c(730, 227, 1242),
  weight = c(1000, 1000, 1000)
))
plays <- rbind(plays, data.frame(
  userID = c(9999, 9999, 9999, 9999),
  artistID = c(210, 1116, 220, 207),
  weight = c(1000, 1000, 1000, 1000)
))

#Creating the realRatingMatrix
rrm <- as(plays,"realRatingMatrix")
rrm

#Evaluating UBCF
srrm <- rrm[rowCounts(rrm) >= 5,]
es <- evaluationScheme(srrm, method="split", train=0.9, given=3, goodRating=0.001)
ev <- evaluate(es, "UBCF", parameter = list(nn=750), type="ratings", 
               n=10, progress=FALSE)
avg(ev)

#checking UBCF results
rec <- Recommender(rrm, method = "UBCF", parameter = list(nn=750))
pre <- predict(rec, rrm["9999"], n=10)
data.frame(artistID = as.integer(as(pre, "list")[[1]])) %>% 
  left_join (artists, by="artistID")
pre <- predict(rec, rrm["9998"], n=10)
data.frame(artistID = as.integer(as(pre, "list")[[1]])) %>% 
  left_join (artists, by="artistID")

#Evaluating statistics for multiple values of nn
nns <- seq(50, 1000, 50)
statx <- sapply(nns, function(nnx){
  ev <- evaluate(es, "UBCF", parameter = list(nn=nnx), n=10,
                 progress = FALSE)
  return(avg(ev))
})
qplot(nns, statx[1,], main="True Positives") #TP
qplot(nns, statx[2,], main="False Positives") #FP
qplot(nns, statx[3,], main="False Negatives") #FN
qplot(nns, statx[4,], main="True Negatives") #TN
qplot(nns, statx[5,], main="Precision") #precision
qplot(nns, statx[6,], main="Recall") #recall
qplot(nns, statx[7,], main="TPR") #TPR
qplot(nns, statx[8,], main="FPR") #FPR

#True Positive vs True Negative
plot(nns, statx[1,])
par(new=TRUE)
plot(nns, statx[4,],col="green")

#Running UBCF with nn=300
rec <- Recommender(rrm, method = "UBCF", parameter = list(nn=300))
pre <- predict(rec, rrm["9999"], n=10)
data.frame(artistID = as.integer(as(pre, "list")[[1]])) %>% 
  left_join (artists, by="artistID")
pre <- predict(rec, rrm["9998"], n=10)
data.frame(artistID = as.integer(as(pre, "list")[[1]])) %>% 
  left_join (artists, by="artistID")

#Running UBCF from a function
get_suggestions <- function(id) {
  nnx = 400
  resp = NULL
  while (nnx<2000) {
    rec <- Recommender(rrm, method = "UBCF", parameter = list(nn=nnx))
    pre <- predict(rec, rrm[ toString(id) ], n=10)
    if (length(as(pre, "list")[[1]])<10 || nnx>=1950){
      return(data.frame(artistID = as.integer(as(resp, "list")[[1]])) %>% 
               left_join (artists, by="artistID"))
    } else {
      resp = pre
    }
    nnx=nnx+50
  }
}
get_suggestions(9999)
get_suggestions(9998)
