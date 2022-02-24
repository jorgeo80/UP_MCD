library(tidyverse)
library(rtweet)
library(igraph)
# ===================
library(wordcloud) 
library(tidytext) 
library(tokenizers) 
library(viridis) 

get_trends('Mexico')$trend
df_tuits <- search_tweets(q="#PeorQueMegacable", 
                          n = 5000, 
                          type="mixed",
                          lan='es')
wordcloud(df_tuits$text)

df_tokens <- tibble(message = 1:nrow(df_tuits), 
                    text = str_replace(df_tuits$text, 
                                       'https://.{1,}\\s{0,1}', '')) %>%
               tidytext::unnest_tokens(word, text) %>%
               count(word, sort = TRUE)


wordcloud(words = df_tokens$word, freq = df_tokens$n)

sw <- search_tweets(q = "lang:es",
                    n = 8000,
                    encoding = 'UTF-8',
                    include_rts = FALSE,
                    retryonratelimit = TRUE,
                    lang = 'es')

stopwords.spanish <-
  tokenize_words(sw$text) %>%
    unlist()  %>%
    table()

stopwords <- 
  tibble(word = names(stopwords.spanish), 
         n = stopwords.spanish) %>% 
    arrange(-n) %>%
    head(n = 0.01 * nrow(.))

mylist <- c(tolower(unique(unlist(df_tuits$mentions_screen_name))),
            'peorquemegacable')

df_wc <-
  tibble(message = 1:nrow(df_tuits), 
         text = str_replace(df_tuits$text, 
                            'https://.{1,}\\s{0,1}', '')) %>%
    tidytext::unnest_tokens(word, text) %>%
    anti_join(stopwords, by = 'word') %>%
    count(word, sort = TRUE)  %>%
    filter(!word %in% mylist)

wordcloud(words = df_wc$word, 
          freq = df_wc$n,
          random.order = FALSE,
          max.words = 500,
          colors = brewer.pal(8,'Dark2')
          )



