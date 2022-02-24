library(tidyverse)
library(rtweet)
library(igraph)

setwd('~/Documents/academia/UP_MCD/Materias/RedesSociales')
getwd()

karate_club <- read.csv('karate_club.csv', header = TRUE)
karate_node <- read.csv('karate_node_labels.csv', header = TRUE)
head(karate_node)

g <-
  karate_club %>%
    graph_from_data_frame() %>%
    as.undirected()

V(g)
get.data.frame(g, what = 'vertices') %>% head()
V(g)$club <- karate_node$labelId 

set.seed(1010)
plot(g,
     vertex.size = 10,
     vertex.color = V(g)$club,
     vertex.label = NA
     )

# =========================================================
coms <- cluster_edge_betweenness(g)
coms$membership
membership(coms)

V(g)$cluster <- coms$membership

get.data.frame(g, what = 'vertices')

set.seed(1010)
plot(g,
     vertex.size = 10,
     vertex.color = V(g)$cluster,
     vertex.label = NA
)

# =========================================================
coms <- cluster_louvain(g)
coms$membership
membership(coms)

V(g)$cluster_louvain <- coms$membership

get.data.frame(g, what = 'vertices')

set.seed(1010)
plot(g,
     vertex.size = 10,
     vertex.color = V(g)$cluster_louvain,
     vertex.label = NA
)

# =========================================================
coms <- cluster_label_prop(g)
coms$membership
membership(coms)

V(g)$cluster_label <- coms$membership

get.data.frame(g, what = 'vertices')

set.seed(1010)
plot(g,
     vertex.size = 10,
     vertex.color = V(g)$cluster_label,
     vertex.label = NA
)


# =========================================================
get.data.frame(g, what = 'vertices') %>%
  mutate(diff_cluster = ifelse(club == cluster, 1, 0),
         diff_louvain = ifelse(club == cluster_louvain, 1, 0),
         diff_label = ifelse(club == cluster_label, 1, 0)) %>%
  summarise(cluster_per = sum(diff_cluster) / n(),
            louvain_per = sum(diff_louvain) / n(),
            label_per = sum(diff_label) / n())


# =========================================================
get_trends('Mexico')$trend

tuits <- search_tweets(q="AMLO", n = 15000, type="mixed")

g_tweets <- 
  tuits %>%
    filter(retweet_count > 0) %>%
    select(screen_name, mentions_screen_name) %>%
    unnest(mentions_screen_name) %>%
    na.omit() %>%
    graph_from_data_frame()

g_tweets

# Grupo de nodos mayor a 10
keep <- V(g_tweets)[degree(g_tweets) > 10]
g_tweets_v1 <- induced_subgraph(g_tweets, keep)
g_tweets_v1 <- as.undirected(g_tweets_v1)

# Quita los auto enlaces
g_tweets_v1 <- igraph::simplify(g_tweets_v1)

# 
g_tweets_v1 <- delete_vertices(g_tweets_v1, degree(g_tweets_v1) == 0)

set.seed(1010)
plot(g_tweets_v1,
     vertex.size = 2,
     vertex.color = 'black',
     vertex.label = NA
)

# == Newman
coms <- cluster_edge_betweenness(g_tweets_v1)
coms$membership
V(g_tweets_v1)$Cluster_Newman <- coms$membership

set.seed(1010)
plot(g_tweets_v1,
     vertex.size = 2,
     vertex.color = V(g_tweets_v1)$Cluster_Newman,
     vertex.label = NA
)

# == Louvain
coms <- cluster_louvain(g_tweets_v1)
coms$membership
V(g_tweets_v1)$Cluster_Louvain <- coms$membership

set.seed(1010)
plot(g_tweets_v1,
     vertex.size = 2,
     vertex.color = V(g_tweets_v1)$Cluster_Louvain,
     vertex.label = NA
)

# == Label Propagation
coms <- cluster_label_prop(g_tweets_v1)
coms$membership
V(g_tweets_v1)$Cluster_Label <- coms$membership

set.seed(1010)
plot(g_tweets_v1,
     vertex.size = 2,
     vertex.color = V(g_tweets_v1)$Cluster_Label,
     vertex.label = NA
)


# Otros algoritmos que se pueden usar 
# infomap, walktrap, fast_greedy, leading_eigen

# se puede agregar peso a las aristas para mejorar sus particiones
get.edgelist(g_tweets_v1)
get.data.frame(g_tweets_v1, what = 'edges') %>% 
  head()


# =================================================================
membership.frame <- tibble(name=V(g_tweets_v1)$name,
                           community=V(g_tweets_v1)$Cluster_Louvain) 
edge.weight <- 10 
edge.list <- get.edgelist(g_tweets_v1) 
weights <- c() 
for(i in 1:nrow(edge.list)){ 
  com.1 <- membership.frame[membership.frame$name == edge.list[i,][1],]$community 
  com.2 <- membership.frame[membership.frame$name == edge.list[i,][2],]$community 
  if(com.1 == com.2){ 
    weight <- edge.weight 
    }else{ 
      weight <- 1 
      } 
  weights <- c(weights,weight) 
} 
E(g_tweets_v1)$weight <- weights 


# Monitoreo de una marca o slogan en redes sociales
# Estudio sobre la polarización, que grupos se forman
# Analisis de los metadatos de los usuarios
# Redes de concurrencia sobre conversaciones de interes
# algoritmos de calsificacion para detectar bots
# analisis de interaccion entre personas de una misma conmunidad
# estudiar la viralizacion de una campaña











