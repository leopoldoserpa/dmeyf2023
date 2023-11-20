# para correr el Google Cloud
#   8 vCPU
#  64 GB memoria RAM


# limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

require("data.table")
require("lightgbm")

#Para imputacion
install.packages("mice")
library(mice)
library(dplyr)

# defino los parametros de la corrida, en una lista, la variable global  PARAM
#  muy pronto esto se leera desde un archivo formato .yaml
PARAM <- list()
#INDICAR METODO DE IMPUTACION
metodo_imputacion <- "mice_RF_1pct_m1_m5"

PARAM$experimento <- paste("KA8240",metodo_imputacion,sep="_")

PARAM$input$dataset <- "./datasets/competencia_03.csv.gz"

# meses donde se entrena el modelo
PARAM$input$training <- c(202010,202011,202012, 202101, 202102, 202103,202104,202105)
PARAM$input$future <- c(202107) # meses donde se aplica el modelo

PARAM$finalmodel$semilla <- c(290497, 540187, 987851, 984497, 111893, 100103, 100189, 101987, 991981, 991987,106853,
                              191071,337511,400067,991751,729191,729199,729203,729217,729257)

# hiperparametros intencionalmente NO optimos
PARAM$finalmodel$optim$num_iterations <- 10892
PARAM$finalmodel$optim$learning_rate <- 0.098899556
PARAM$finalmodel$optim$feature_fraction <- 0.682425846
PARAM$finalmodel$optim$min_data_in_leaf <- 24341
PARAM$finalmodel$optim$num_leaves <- 960

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui empieza el programa
setwd("~/buckets/b1")

# cargo el dataset donde voy a entrenar
dataset <- fread(PARAM$input$dataset, stringsAsFactors = TRUE)


# Catastrophe Analysis  -------------------------------------------------------

# zero ratio imputo nulos

# Initialize an empty list to store the results
results_list <- list()
# Loop through each field in 'campos_buenos'
# los campos que se van a utilizar
campos_buenos <- setdiff(colnames(dataset), c("clase_ternaria", "clase01"))

for (campo in campos_buenos) {
  # Calculate the zero ratio for the current field and group by 'foto_mes'
  result <- dataset[, .(zero_ratio = sum(get(campo) == 0, na.rm = TRUE) / .N), by = foto_mes]
  
  # Add the result to the list
  results_list[[campo]] <- result
}

#FIltro aquellas variables donde zero_ratio es =1

# Initialize an empty list to store selected variables
selected_variables <- list()

# Loop through each field in 'campos_buenos'
for (campo in names(results_list)) {
  # Filter the data.table for rows where 'zero_ratio' is equal to 1
  filtered_result <- results_list[[campo]][zero_ratio == 1]
  
  # If any rows meet the condition, add the field name to the list
  if (nrow(filtered_result) > 0) {
    selected_variables[[campo]] <- filtered_result
  }
}

# 'selected_variables' contains variables with 'zero_ratio' equal to 1
selected_variables

#Trato usar esto como filtro

# Initialize a list to store filter conditions for each selected variable
filter_conditions <- list()

# Loop through each selected variable in 'selected_variables'
for (variable_name in names(selected_variables)) {
  # Get the 'foto_mes' values for which 'zero_ratio' is 1
  filtered_foto_mes <- selected_variables[[variable_name]]$foto_mes
  
  # Create a filter condition for the current variable and 'foto_mes'
  filter_condition <- dataset$foto_mes %in% filtered_foto_mes
  
  # Add the filter condition to the list
  filter_conditions[[variable_name]] <- filter_condition
}

# Combine filter conditions using logical OR (|) to filter the dataset
combined_filter <- Reduce(`|`, filter_conditions)

# Apply the combined filter to the dataset
filtered_dataset <- dataset[combined_filter]

#Coloco a todas las variables que cumplen el combined_filter NA

dataset[combined_filter, names(selected_variables) := NA]

#------------------------------------------------------------------------------
#Imputación de nulos

#1% de los continuan en el periodo de entrenamiento
continua_training_0.01 <- sample_frac(dataset[dataset$foto_mes %in% c(202010,202011,202012, 202101, 202102, 202103,202104,202105)
                                             & clase_ternaria == "CONTINUA"], 0.01, seed = PARAM$lgb_semilla)

#100% BAJA+1 Y BAJA+2
no_continua_training <- dataset[dataset$foto_mes %in% c(202010,202011,202012, 202101, 202102, 202103,202104,202105) 
                                & clase_ternaria != "CONTINUA"]

training_subsampling_continua <- rbind(continua_training_0.01, no_continua_training)

time_imp.train <- system.time({imp.train <- mice(
  data = training_subsampling_continua[,..campos_buenos], 
  method = 'rf', 
  seed = PARAM$lgb_semilla, 
  #n.core = 3, 
  m = 1,
  maxit = 5,
  printFlag = TRUE
  #verbose = TRUE,
  #ignore = ignored
)})


time_imp.future <-  system.time({imp.future <- mice.mids(imp.train, newdata = dataset[foto_mes %in% c(202107),..campos_buenos],
                                                  maxit=5,printFlag = T)})

time_imp.train_full <- system.time({imp.train_full <- mice.mids(imp.train,newdata = dataset[foto_mes %in% c(202010,202011,202012, 202101, 202102, 202103,202104,202105),..campos_buenos], 
                                               maxit=5,printFlag = T)})

# Chequeo nulos
sum(is.na(dataset[foto_mes %in% c(202010, 202011, 202012, 202101, 202102, 202103, 202104, 202105,202107)]))

# Reemplazo 
dataset[foto_mes %in% c(202010,202011,202012, 202101, 202102, 202103,202104,202105),campos_buenos] <- complete(imp.train_full)
dataset[foto_mes %in% c(202107),campos_buenos] <- complete(imp.future)


# Chequeo que no hay nulos
sum(is.na(dataset[foto_mes %in% c(202010, 202011, 202012, 202101, 202102, 202103, 202104, 202105, 202107)]))

#------------------------------------------------------------------------------
# LAGS 1, 3 y 6 ################################################################

#calculate lags 1,3 and 6 in campos_buenos
dataset[, paste0(campos_buenos, "_lag1") := lapply(.SD, shift, 1L, type = "lag"), .SDcols = campos_buenos]
dataset[, paste0(campos_buenos, "_lag3") := lapply(.SD, shift, 3L, type = "lag"), .SDcols = campos_buenos]
dataset[, paste0(campos_buenos, "_lag6") := lapply(.SD, shift, 6L, type = "lag"), .SDcols = campos_buenos]

################################################################################

#--------------------------------------

# paso la clase a binaria que tome valores {0,1}  enteros
# set trabaja con la clase  POS = { BAJA+1, BAJA+2 }
# esta estrategia es MUY importante
dataset[, clase01 := ifelse(clase_ternaria %in% c("BAJA+2", "BAJA+1"), 1L, 0L)]

#--------------------------------------

# los campos que se van a utilizar
campos_buenos <- setdiff(colnames(dataset), c("clase_ternaria", "clase01"))

#--------------------------------------


# establezco donde entreno
dataset[, train := 0L]
dataset[foto_mes %in% PARAM$input$training, train := 1L]

#--------------------------------------
# creo las carpetas donde van los resultados
# creo la carpeta donde va el experimento
dir.create("./exp/", showWarnings = FALSE)
dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)

# Establezco el Working Directory DEL EXPERIMENTO
setwd(paste0("./exp/", PARAM$experimento, "/"))



# dejo los datos en el formato que necesita LightGBM
dtrain <- lgb.Dataset(
  data = data.matrix(dataset[train == 1L, campos_buenos, with = FALSE]),
  label = dataset[train == 1L, clase01]
)

# aplico el modelo a los datos sin clase
dapply <- dataset[foto_mes == PARAM$input$future]

# genero la tabla de entrega
tb_entrega <- dapply[, list(numero_de_cliente, foto_mes,clase_ternaria,clase01)]

for (semilla in PARAM$finalmodel$semilla){
  
  cat(paste0("\nArrancamos con la semilla ",semilla))
  
  # Hiperparametros FIJOS de  lightgbm
  PARAM$finalmodel$lgb_basicos <- list(
    boosting = "gbdt", # puede ir  dart  , ni pruebe random_forest
    objective = "binary",
    metric = "custom",
    first_metric_only = TRUE,
    boost_from_average = TRUE,
    feature_pre_filter = FALSE,
    force_row_wise = TRUE, # para reducir warnings
    verbosity = -100,
    max_depth = -1L, # -1 significa no limitar,  por ahora lo dejo fijo
    min_gain_to_split = 0.0, # min_gain_to_split >= 0.0
    min_sum_hessian_in_leaf = 0.001, #  min_sum_hessian_in_leaf >= 0.0
    lambda_l1 = 0.0, # lambda_l1 >= 0.0
    lambda_l2 = 0.0, # lambda_l2 >= 0.0
    max_bin = 31L, # lo debo dejar fijo, no participa de la BO
    
    bagging_fraction = 1.0, # 0.0 < bagging_fraction <= 1.0
    pos_bagging_fraction = 1.0, # 0.0 < pos_bagging_fraction <= 1.0
    neg_bagging_fraction = 1.0, # 0.0 < neg_bagging_fraction <= 1.0
    is_unbalance = FALSE, #
    scale_pos_weight = 1.0, # scale_pos_weight > 0.0
    
    drop_rate = 0.1, # 0.0 < neg_bagging_fraction <= 1.0
    max_drop = 50, # <=0 median no limit
    skip_drop = 0.5, # 0.0 <= skip_drop <= 1.0
    
    extra_trees = TRUE, # Magic Sauce
    
    seed = semilla
  )
  
  # genero el modelo
  param_completo <- c(PARAM$finalmodel$lgb_basicos,
    PARAM$finalmodel$optim)
  
  modelo <- lgb.train(
    data = dtrain,
    param = param_completo,
  )
  
  #--------------------------------------
  # ahora imprimo la importancia de variables
  tb_importancia <- as.data.table(lgb.importance(modelo))
  archivo_importancia <- paste0(semilla,"_impo.txt")
  
  fwrite(tb_importancia,
    file = archivo_importancia,
    sep = "\t"
  )
  
  #--------------------------------------
  
  

  # aplico el modelo a los datos nuevos
  prediccion <- predict(
    modelo,
    data.matrix(dapply[, campos_buenos, with = FALSE])
  )
  
  tb_entrega[, paste0("prob_",semilla) := prediccion]

}
# grabo las probabilidad del modelo
fwrite(tb_entrega,
  file = "prediccion.csv")

# ordeno por probabilidad descendente
#setorder(tb_entrega, -prob)


# genero archivos con los  "envios" mejores
# deben subirse "inteligentemente" a Kaggle para no malgastar submits
# si la palabra inteligentemente no le significa nada aun
# suba TODOS los archivos a Kaggle
# espera a la siguiente clase sincronica en donde el tema sera explicado

#cortes <- seq(8000, 15000, by = 500)
#for (envios in cortes) {
#  tb_entrega[, Predicted := 0L]
#  tb_entrega[1:envios, Predicted := 1L]

  #fwrite(tb_entrega[, list(numero_de_cliente, Predicted)],
  #  file = paste0(PARAM$experimento, "_", envios, ".csv"),
  #  sep = ","
  #)
#}

cat("\n\nLa generacion de los archivos para Kaggle ha terminado\n")
