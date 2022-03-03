library(tidyverse)
library(rtweet)
library(tidytext) 
library(tokenizers) 

url <- "https://raw.githubusercontent.com/Anishka0107/Sentiment-Analyzer/master/movie-pang02.csv"

df <- read.csv(url, stringsAsFactors=FALSE) 

df <- tibble(document = 1:nrow(df), class = df$class, text = df$text)

df$class <- as.factor(df$class)

df <- as.tibble(df)

head(df)

df %>% count(class)

#sw <- search_tweets("lang:en", n=8000, include_rts = FALSE,
#                    encoding= 'UTF-8', retryonratelimit = TRUE)

stopwords.english <-
  sw$text %>% 
    tokenize_words() %>%
    unlist() %>% 
    table()

stopwords.english <- 
  names(head(sort(stopwords.english, decreasing = TRUE), n = 50))

tidy.df <-
  df %>% 
    unnest_tokens(word,text) %>%
    filter(!word %in% stopwords.english)

pos.words <-
  tidy.df %>%
    filter(class %in% 'Pos') %>%
    pull(word)

neg.words <-
  tidy.df %>%
    filter(class %in% 'Neg') %>%
    pull(word)

calc.probs <- function(x){
  counts <- table(x) + 1
  log(counts / sum(counts))
}

pos.probs <- calc.probs(pos.words)
neg.probs <- calc.probs(neg.words)

calc.probs.rare <- function(x){
  counts <- table(x) + 1
  log(1 / sum(counts))
}

pos.probs.rare <- calc.probs.rare(pos.words)
neg.probs.rare <- calc.probs.rare(neg.words)

calc.sentiment <- function(text, stopwords){
  
  test <-
    tibble(line = 1, text = text) %>%
      unnest_tokens(word, text) %>%
      filter(!word %in% stopwords) %>%
      pull(word)
  
  prob.pos <- sum(pos.probs[test], na.rm = TRUE) + 
    sum(is.na(pos.probs[test])) * pos.probs.rare
  
  prob.neg <- sum(neg.probs[test], na.rm = TRUE) + 
    sum(is.na(neg.probs[test])) * neg.probs.rare
  
  ifelse(prob.pos > prob.neg, 'Pos', 'Neg')
  
}

calc.sentiment('I am very happy', stopwords.english)


#df_tweets <- search_tweets("#Ukranie", n=8000, include_rts = FALSE,
#                           encoding= 'UTF-8', retryonratelimit = TRUE,
#                           lang = 'en')

#install.packages('tictoc')
#library(tictoc)
#tic()
df_tweets_class <- NULL
for (i in 1:nrow(df_tweets)){
  tweet <- df_tweets$text[i]
  df_tweets_class <- rbind(df_tweets_class,
                           cbind(class = calc.sentiment(tweet, 
                                                        stopwords.english), 
                                  text = tweet))
}
#toc()
as.tibble(df_tweets_class)


install.packages('syuzhet')
library(syuzhet) # para analisis de sentimiento





