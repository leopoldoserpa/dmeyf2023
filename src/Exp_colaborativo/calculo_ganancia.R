require("data.table")

df <- fread("exp_KA8240_no_imputar_prediccion.csv")

POS_ganancia <- 273000
NEG_ganancia <- -7000

mapeo <- function(corte,df){
  print(paste0("corte ",corte))
  df[,Predicted:=0L]
  df[1:corte,Predicted:=1L]
  
  #df[,ganancia:=ifelse(Predicted==1&clase01==1, POS_ganancia, NEG_ganancia)]
  #G <- sum(df$ganancia)
  
  conf_matrix <- table(df$Predicted,df$clase01)
  TP <- conf_matrix[2,2]
  TN <- conf_matrix[1,1]
  FP <- conf_matrix[2,1]
  FN <- conf_matrix[1,2]
  
  G <- TP*POS_ganancia + FP*NEG_ganancia
  return(G)
}

curva_ganancia <- function(df,semilla){
  print(paste0("Semilla: ",semilla))
  df <- df[order(-get(paste0("prob_",semilla))),]
  cortes <- 1:nrow(df)
  
  ganancias <- c()
  cortes <- seq(1,100000,1000)
  ganancias <- map_dbl(cortes,mapeo,df)
  
  return(ganancias)
}

semillas <- gsub("prob_","",names(df)[grepl("prob",names(df))])
curvas <- data.table("corte"=seq(1,100000,1000))
for(semilla in semillas){
  curvas[,semilla] <- curva_ganancia(df,semilla)
}

#make a line plot grouped by "semilla"
require("ggplot2")
curvas[corte<100000,] |> 
  melt(id.vars = "corte",variable.name = "semilla",value.name = "ganancia") |> 
ggplot(aes(x=corte,y=ganancia,color=semilla))+
  geom_line()+
  theme_bw()+
  labs(title="Curva de ganancia",
       subtitle="Ganancia acumulada en funcion del numero de clientes",
       x="Numero de clientes",
       y="Ganancia acumulada",
       color="Semilla")

G <- unlist(lapply(curvas[corte>14000&corte<27000],mean))
