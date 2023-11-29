#setwd('~/buckets/b1')

setwd("C:/Users/leopo/Desktop/Maestria DATA MINING/DMEyF/KAGGLE 3RA COMPETENCIA")
require("data.table")

# pengue <- fread("exp/KA8240_03_pengue_marchesini/prediccion.txt")
# casalli <- fread("exp/final_casalli/prediccion.txt")
# serpa <- fread('exp/KA8240_serpa/prediccion.txt')
# flores <- fread('exp/KA5240_vanesa_flores/prediccion.txt')
# lags <- fread('exp/exp_final/V4_2_prob.csv')[,list(numero_de_cliente,foto_mes,prob)]
#ensamble_2 <- fread('exp/KA8240_ensamble_2_LRalto_neg_bagging_FILAS/prediccion.csv')
#ensamble_5 <- fread('exp/KA8240_ensamble_5/prediccion.csv')
#ensamble_1 <- fread("KA8240_ensamble_1/prediccion.csv")
#ensamble_2 <- fread('KA8240_ensamble_2_LRalto_neg_bagging_FILAS/prediccion.csv', drop = c("numero_de_cliente","foto_mes"))
ensamble_3 <- fread("exp/KA8240_ensamble_3/prediccion.csv")
#ensamble_4 <- fread("KA8240_ensamble_4/prediccion.csv", drop = c("numero_de_cliente","foto_mes"))
ensamble_5 <- fread('exp/KA8240_ensamble_5/prediccion.csv', drop = c("numero_de_cliente","foto_mes"))

# Concatenar los ensambles
df <- cbind(ensamble_3, ensamble_5)

# Calcular el promedio
df[, prob_avg := rowMeans(df[, .SD, .SDcols = -(1:2)], na.rm = TRUE)]

# Seleccionar las columnas necesarias
columnas_necesarias <- c("numero_de_cliente", "prob_avg")

# Mostrar el resultado
df <- df[, ..columnas_necesarias]

#Ordeno por probabilidad descendente
setorder(df,-prob_avg)


#Creo las predicciones para Kaggle
dir.create("exp/ensamble_modelos")
cortes <- seq(8000, 15000, by = 500)
for (envios in cortes) {
  df[, Predicted := 0L]
  df[1:envios, Predicted := 1L]
  
  fwrite(df[, list(numero_de_cliente, Predicted)],
         file = paste0("exp/ensamble_modelos/ensamble_3_5", "_", envios, ".csv"),
         sep = ","
  )
}
