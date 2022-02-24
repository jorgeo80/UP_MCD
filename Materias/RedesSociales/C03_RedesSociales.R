library(tidyverse)
library(rtweet)

get_trends('Mexico') %>% View()
search_tweets(q="Panamá filter:news", n=10) %>% View()


search_tweets(q = "Mejores ofertas", 
              lang = 'es',
              since = '2022-01-30', 
              until = '2022-01-31', 
              type = 'mixed') %>% View()

# == Esta función es para guardar tablas de twitters [write_as_csv()]
# == Y esta función es para leer las tablas guardadads [read_twitter_csv()]


# == para descargar información de cuentas en particular
# == get_timeline()

get_timeline('UPMexico', n=100) %>% View()


# == ggplot2
df_tweets <- search_tweets(q = "Lozano", lang = 'es', type = 'mixed')

df_tweets %>% 
  count(screen_name) %>%
  arrange(-n)

# =========================================================================== #
#                         ========= Tarea =========                           #
# Buscar un hastag o de una cuenta de interes, descargar 15k tweets           #
# serie de tiempo de tweets vs retweets                                       #  
# instalar igraph                                                             #
# =========================================================================== #




