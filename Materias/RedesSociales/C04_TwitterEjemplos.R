# install.packages('igraph')
library(tidyverse)
library(rtweet)
library(igraph)

# Ejemplo de red dirigida
g1 <- graph(edges = c(1, 2, 2, 3, 3, 1))
plot(g1)

# Ejemplo de red no dirigida
g2 <- graph(edges = c(1, 2, 2, 3, 3, 1), directed = FALSE)
plot(g2)

# Con esta sentencia podemos saber si es dirigida o no la red
is.directed(g1)
# Con este podemos los vertices de la red (vertex)
V(g1)
# Con este podemos las aristas de la red (edges)
E(g1)


# Ejemplo
g3 <- graph(edges = c('Obama', 'Rihanna', 'Rihanna', 'Bieber', 'Bieber', 
                      'Obama', 'Perry', 'Bieber'))
plot(g3)


# Ejemplo
g4 <- graph(edges = c('Obama', 'Rihanna', 'Rihanna', 'Bieber', 'Bieber', 
                      'Obama', 'Perry', 'Bieber'),
            isolates = c('CR7', 'Gaga'))
plot(g4)


# Atributos de la red (Vertices)
get.data.frame(g3, what = 'vertices')
V(g3)$name

# Atributos de la red (Aristas)
get.data.frame(g4, what = 'edges')
E(g4)

plot(g4, 
     edge.arrow.size = 0.5,
     vertex.color = 'purple',
     vertex.frame.color = 'purple',
     vertex.label.cex = 0.8,
     vertex.label.dist = 3,
     vertex.size = 25,
     vertex.label.color = 'orange',
     )

degree(g4)
degree(g4, mode = 'in')
degree(g4, mode = 'out')
degree(g4, mode = 'all')

# Intermediación
betweenness(g4)
betweenness(g4, directed = FALSE)

# Coeficiente de centralidad
closeness(g4)
closeness(g4, mode = 'in')
closeness(g4, mode = 'out')
closeness(g4, mode = 'all')

df_g4 <-
get.data.frame(g4, what = 'vertices') %>%
  mutate(degree_in = degree(g4, mode = 'in'),
         degree_out = degree(g4, mode = 'out'),
         degree_all = degree(g4, mode = 'all'),
         closeness_in = closeness(g4, mode = 'in'),
         closeness_out = closeness(g4, mode = 'out'),
         closeness_all = closeness(g4, mode = 'all'),
         gender = c('M', 'F', 'M', 'F', 'M', 'F'))

df_g4 %>%
ggplot(aes(x = reorder(name, degree_all), y = degree_all)) +
  geom_bar(stat = 'identity', 
           fill = ifelse(df_g4$gender == 'M', 'steelblue', 'coral')) +
  labs(xlab = 'User',
       tittle = 'Degree of toy network users',
       subtitle = 'Excercise for showing how to visualize degrees') +
  theme_bw()

# Ejemplo Twitter

df_tweets <- search_tweets(q='#TodosSomosHugo', n=5000) %>%
  as_tibble()
head(df_tweets)

g <-
df_tweets %>%
  filter(retweet_count >= 1) %>%
  select(screen_name, mentions_screen_name) %>%
  unnest(mentions_screen_name) %>%
  na.omit()  %>%
  graph_from_data_frame()

# Investigar GEPH

V(g)
E(g)
hist(degree(g))
hist(log(degree(g)))

# plot(g)
# Cuando es muy grande, es mejor tomar el nucleo de la red
# que es una subred inducida hasta el grado de selección

core <- V(g)[degree(g) > 10]
g2 <- induced_subgraph(g, core)

#
plot(g2,
     vertex.size=2,
     vertex.color='black',
     edge.arrow.size=0,
     vertex.label=NA
     )


plot(igraph::simplify(g2),
     vertex.size=2,
     vertex.color='black',
     edge.arrow.size=0,
     vertex.label=NA
)

g3 <-
delete.vertices(igraph::simplify(g2), 
                degree(igraph::simplify(g2)) == 0)


plot(g3,
     layaout = layout_as_star,
     vertex.size=2,
     vertex.color='black',
     edge.arrow.size=0,
     vertex.label=NA
)

# ============================================================================ #
# === Pequeño script para ver qué hacen diferentes layouts. 
# ============================================================================ #

layouts <- c("layout_nicely", "layout_with_kk", "layout_in_circle", 
             "layout_on_grid", "layout_on_sphere", "layout_randomly", 
             "layout_with_fr", "layout_as_star", "layout_with_gem") 
par(mfrow=c(3,3),mar=c(1,1,1,1)) 
for(i in 1:length(layouts)){ 
  LO <- layouts[i] 
  l <- do.call(LO, list(g3)) 
  plot(g3, 
       layout = l, 
       vertex.color = "blue", 
       vertex.size = 2, 
       edge.arrow.size = 0, 
       vertex.label = NA, 
       main = LO
       ) 
  } 

