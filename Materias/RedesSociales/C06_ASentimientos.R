# ===================
# install.packages('wordcloud') 
# install.packages('tidytext') 
# install.packages('tokenizers') 
# install.packages('viridis')
# install.packages('RColorBrewer')
# install.packages('viridisLite')
# ===================
library(tidyverse)
library(rtweet)
library(igraph)
# ===================
library(wordcloud) 
library(tidytext) 
library(tokenizers) 
library(viridis) 

get_trends('Mexico')$trend
df_tuits_mrbl <- search_tweets(q="Mr.Blanco's", 
                                n = 15000, 
                                type="mixed",
                                lan='es')
df_tuits_mrbl

g_tweets <- 
  tuits %>%
  filter(retweet_count > 0) %>%
  select(screen_name, mentions_screen_name) %>%
  unnest(mentions_screen_name) %>%
  na.omit() %>%
  graph_from_data_frame()

g_tweets
