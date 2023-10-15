# limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

# Load the data.table library if not already loaded
require(data.table)

setwd("~/buckets/b1")

# Cargo el dataset
dataset <- fread("./datasets/competencia_02.csv.gz")

#dataset 1 tiene aquellos clientes "BAJA+2"
dataset1 <- (dataset[dataset$clase_ternaria=='BAJA+2'])

#Numero de clientes que se dieron de baja
cat('clientes_BAJA+2 =', nrow(dataset1))

#dataset 2 tiene el historial en el banco de los clientes "BAJA+2"
dataset2 <- dataset[dataset$numero_de_cliente %in% dataset1$numero_de_cliente]

#Historial de los BAJA+2
cat('clientes_BAJA+2 =', nrow(dataset2))


#Clustering
install.packages("randomForest")
library(randomForest)

names(dataset1)[colSums(is.na(dataset1))>0]

mean(dataset$mtarjeta_visa_descuentos, na.rm=TRUE)

#Reemplazo los na
dataset1 <- na.roughfix(dataset1)

for (col_name in names(dataset1)[colSums(is.na(dataset1))>0]){
    dataset1
}

names(dataset1)
dataset1[mtarjeta_visa_descuentos == 44.70201, ]

#Calculo el clustering
rf.fit <- randomForest(dataset1, y = NULL, ntree = 1000, proximity = TRUE, oob.prox = TRUE, seed = 100103, na.action = na.roughfix)

hclust.rf <- hclust(as.dist(1-rf.fit$proximity), method = "ward.D2")
rf.cluster = cutree(hclust.rf, k=7)

dataset1.pc$rf.clusters <- rf.cluster




#Ordeno al reves
setorder(dataset2, -foto_mes, -numero_de_cliente)

# Set the reference month for each 'numero_de_cliente' in a data.table
reference_months <- dataset2[, .(numero_de_cliente, reference_month = max(foto_mes)), by = numero_de_cliente]

# Merge the reference_months data.table with the original dataset
dataset3 <- merge(dataset2, reference_months[, .(numero_de_cliente, reference_month)], by = "numero_de_cliente")

# Calculate the differences
dataset3[, c("difference_1") := .(foto_mes - reference_month)]

dataset3

dataset3[dataset3$numero_de_cliente==29199686, .(foto_mes, numero_de_cliente, clase_ternaria, difference_1)]











dataset2



#Murio el 201911
max(dataset1[dataset1$numero_de_cliente==29199686, .(foto_mes)])

#Ultimo mes donde hay registro
max(dataset2[dataset2$numero_de_cliente==29199686, .(foto_mes)])

max(dataset[dataset$numero_de_cliente==29199686, .(foto_mes)])

dataset[dataset$numero_de_cliente==29199686, .(foto_mes, clase_ternaria)]

#Ultimo mes donde hay registro
max(dataset1[dataset1$numero_de_cliente==34486883, .(foto_mes)])

max(dataset2[dataset2$numero_de_cliente==34486883, .(foto_mes)])

max(dataset[dataset$numero_de_cliente==34486883, .(foto_mes)])

dataset[dataset$numero_de_cliente==34486883, .(foto_mes, clase_ternaria)]