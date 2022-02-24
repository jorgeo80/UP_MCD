# Cargamos la bilbioteca que necesitamos
library(tidyverse)

# Descargamos la tabla de datos alojada en una carpeta de Google Drive.
# Usamos encoding = "UTF-8" para leer correctamente acentos y caracteres especiales
id <- "1Ip0riTw6-GBkOvnXOaT-6e1yEHRlg32M"
tuits <- read.csv(sprintf("https://docs.google.com/uc?id=%s&export=download", id),
                  encoding = "UTF-8")

# Convertimos los datos al formato tibble.
tuits <- as_tibble(tuits)

# La funci�n glimpse nos permite echar un vistazo a la tabla
glimpse(tuits)

# Usamos la funci�n select() para seleccionar s�lo columnas de inter�s.
tuits %>%
  select(screen_name,followers_count,friends_count)

# Usamos la funci�n unique() para eliminar filas repetidas.
tuits %>%
  select(screen_name,followers_count,friends_count) %>%
  unique()

# Usamos la funci�n arrange() para ordenar las filas de acuerdo a los valores
# de alguna columna. Agremgamos un signo - para hacerlo en orden descendente.
tuits %>%
  select(screen_name,followers_count,friends_count) %>%
  unique() %>%
  arrange(-followers_count)

tuits %>%
  select(screen_name,followers_count,friends_count) %>%
  unique() %>%
  arrange(-friends_count)

# La funci�n print() nos permite indicar cu�ntas filas queremos ver en
# la pantalla.
tuits %>%
  select(screen_name,followers_count,friends_count) %>%
  unique() %>%
  arrange(-friends_count) %>%
  print(n=50)

# Usamos la funci�n filter() para seleccionar s�lo las filas que satisfacen
# una cierta condici�n.
# Usamos esto para crear una tabla s�lo con los tuits de los "influencers".
tuits %>%
  filter(followers_count > 10000) %>%
  select(screen_name, followers_count,created_at,text) -> tuits.de.influencers

glimpse(tuits.de.influencers)
tuits.de.influencers$text

# La columna verified nos dice si una cuenta est� verificada o no por Twitter.
tuits %>%
  select(screen_name, verified, followers_count) %>%
  unique()

tuits %>%
  filter(verified==TRUE) %>%
  select(text)

# Usamos la funci�n count() para contar los valores �nicos de una columna.
tuits %>%
  count(verified)

# Con esta funci�n podemos buscar las cuentas que m�s veces participaron en la 
# conversaci�n.
tuits %>%
  count(screen_name)

tuits %>%
  count(screen_name) %>%
  arrange(-n)

# Cada fila de la tabla es un tuit, no un usuario. Utilizamos las funciones que hemos visto
# para ver cu�ntas cuentas verificadas y cu�ntas no verificadas hay en esta conversaci�n.
tuits %>%
  select(screen_name, verified) %>%
  unique() %>%
  count(verified)

# Las funciones group_by() y summarise() nos permiten agrupar las filas de la tabla
# de acuerdo a una cierta categor�a y calcular estad�sticas en cada uno de los grupos.
# Es conveniente terminar estas operaciones con la funci�n ungroup().
tuits %>%
  select(screen_name, verified, followers_count) %>%
  unique() %>%
  group_by(verified) %>%
  summarise(mean_followers = mean(followers_count)) %>%
  ungroup()

tuits %>%
  select(screen_name, verified, followers_count, friends_count) %>%
  unique() %>%
  group_by(verified) %>%
  summarise(mean_followers = mean(followers_count),
            mean_friends = mean(friends_count)) %>%
  ungroup()

# La funci�n ifelse() devuelve una respuesta binaria dependiendo de si una cierta
# condici�n l�gica se satisface o no.
x <- seq(1,10)
x
y <- ifelse(x < 5, "chico", "grande")
y

# La funci�n mutate() nos permite crear nuevas columnas.
# Usamos mutate() e ifelse() para crear una columna que nos dice si los usuarios son
# populares o no. 
tuits %>%
  mutate(is_popular = ifelse(followers_count < 10000, FALSE, TRUE)) -> tuits
glimpse(tuits)

# Nos preguntamos si acaso los usuarios populares tuitean m�s, en promedio, que los
# no populares. 
tuits %>%
  select(screen_name,is_popular, statuses_count) %>%
  unique() %>%
  group_by(is_popular) %>%
  summarise(mean_statuses = mean(statuses_count)) %>%
  ungroup()

# De la tabla de tuits extraemos una tabla de usuarios.
tuits %>%
  select(screen_name,verified,followers_count,friends_count,
         favourites_count,statuses_count,listed_count) %>%
  unique() -> usuarios
usuarios

# La funcion ecdf(x) (distribuci�n cumulativa emp�rica) nos da la fracci�n
# de observaciones que son menores a x. 
plot(usuarios$followers_count, ecdf(usuarios$followers_count)(usuarios$followers_count),
     xlab = "log(seguidores)",ylab="ecdf",log="x")

# Proponemos una m�trica para medir la influencia de los usaurios en esta conversaci�n.
# Esta m�trica considera el n�mero de amigos, seguidores, statuses, listas y favoritos
# de cada usuario. 
# Para cada variable y para cada usuario calculamos su ecdf() y luego las sumamos.
# Por construcci�n, esta m�trica (top_score) toma valores entre 0 y 5, donde 
# valores altos corresponden a niveles altos de influencia.
usuarios %>%
  mutate(followers_percentile = ecdf(followers_count)(followers_count),
         friends_percentile = ecdf(friends_count)(friends_count),
         listed_percentile = ecdf(listed_count)(listed_count),
         favourites_percentile = ecdf(favourites_count)(favourites_count),
         statuses_percentile = ecdf(statuses_count)(statuses_count)) %>%
  group_by(screen_name) %>%
  summarise(top_score = followers_percentile + friends_percentile +
              listed_percentile + favourites_percentile + statuses_percentile) %>%
  ungroup() %>% 
  mutate(ranking = rank(-top_score)) -> ranking.de.usuarios

ranking.de.usuarios

ranking.de.usuarios %>%
  arrange(ranking)

# Tomamos una muestra aleatoria de usuarios y graficamos su nivel de influencia.
set.seed(12345)
nombres <- sample(usuarios$screen_name,10)

library(viridis=
          pal <- viridis(10)
        
        ranking.de.usuarios %>%
          filter(screen_name %in% nombres) %>%
          select(screen_name, top_score) -> df
        
        df
        
        par(mar=c(5,10,5,1))                     # m�rgenes = c(bottom, left,top,tight)
        barplot(sort(df$top_score) ,             # variable a visualizar
                names.arg= rev(df$screen_name),  # nombre de cada barra
                horiz=TRUE,                      # barras horizontales
                las=2,                           # orientaci�n de las etiquetas
                col=rev(pal),                    # colores de las barras
                xlab = "score",                  # nombre del eje x
                main="Grado de influencia")      # T�tulo                 