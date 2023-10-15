# para correr el Google Cloud
#   8 vCPU
#  64 GB memoria RAM


# limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

require("data.table")
require("lightgbm")


# defino los parametros de la corrida, en una lista, la variable global  PARAM
#  muy pronto esto se leera desde un archivo formato .yaml
PARAM <- list()
PARAM$experimento <- "KA8240b"

PARAM$input$dataset <- "./datasets/competencia_02.csv.gz"

# meses donde se entrena el modelo
PARAM$input$training <- c(201907,	201908,	201909,	201910,	201911,	201912,	202001,	202002,	202003,	202010,	202011,	202012,	202101,	202102,	202103,	202104,	202105)
PARAM$input$future <- c(202107) # meses donde se aplica el modelo

PARAM$finalmodel$semilla <- 1100189

# hiperparametros intencionalmente NO optimos
PARAM$finalmodel$optim$num_iterations <- 2360
PARAM$finalmodel$optim$learning_rate <- 0.055101382
PARAM$finalmodel$optim$feature_fraction <- 0.153838929
PARAM$finalmodel$optim$min_data_in_leaf <- 14319
PARAM$finalmodel$optim$num_leaves <- 728


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
  max_drop = 50, # <=0 means no limit
  skip_drop = 0.5, # 0.0 <= skip_drop <= 1.0

  extra_trees = TRUE, # Magic Sauce

  seed = PARAM$finalmodel$semilla
)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui empieza el programa
setwd("~/buckets/b1")

# cargo el dataset donde voy a entrenar
dataset <- fread(PARAM$input$dataset, stringsAsFactors = TRUE)


# Catastrophe Analysis  -------------------------------------------------------
# deben ir cosas de este estilo
#   dataset[foto_mes == 202006, active_quarter := NA]
campos_buenos <- setdiff(
  colnames(dataset),
  c("numero_de_cliente", "foto_mes", "clase_ternaria")
)

# Initialize an empty list to store the results
results_list <- list()
# Loop through each field in 'campos_buenos'
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

-------------------------------------------------------------------------
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
# Data Drifting
# por ahora, no hago nada

# Ranking de las variables expresadas en pesos pero sin centrar en cero
diccionario <- fread("~/buckets/b1/datasets/DiccionarioDatos_2023.csv")

features_pesos <- as.character(diccionario[unidad == 'pesos', campo])

# Loop through each column in features_pesos and calculate the rank
for (i in features_pesos) {
  dataset[, (i) := frankv(dataset[[i]], order = 1L, na.last = 'keep', ties.method = "dense")]
}

dataset[, 
        .(foto_mes, numero_de_cliente, mrentabilidad), 
        keyby = .(foto_mes, numero_de_cliente)
][order(-mrentabilidad)]

dataset
# Feature Engineering Historico  ----------------------------------------------
#   aqui deben calcularse los  lags y  lag_delta
#   Sin lags no hay paraiso ! corta la bocha
#   https://rdrr.io/cran/data.table/man/shift.html

#Columnas para el lag (saco numero_de_cliente, foto_mes)
columnas_lag <- names(dataset[, 3:(ncol(dataset)-1)])

# Define the lag range (1 to 6)
lag_range <- 1:6

#Creo columnas con el lag
for (i in lag_range) {
  dataset[, (paste0(columnas_lag, "_lag_", i)) := shift(.SD, n = i, type = "lag"), .SDcols = columnas_lag, by = numero_de_cliente]
}

cols = c('foto_mes', 'numero_de_cliente','mrentabilidad',paste0('mrentabilidad_lag_', lag_range))

dataset[numero_de_cliente==29183981, ..cols]

length(dataset$numero_de_cliente)

# Calculate the differences for all columns in columnas_lag
for (col in columnas_lag) {
  for (i in lag_range) {
    lag_col <- paste0(col, "_lag_", i)
    diff_col <- paste0(col, "_deltalag_", i)
    dataset[, (diff_col) := get(col) - get(lag_col)]
  }
}
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
archivo_importancia <- "impo.txt"

fwrite(tb_importancia,
  file = archivo_importancia,
  sep = "\t"
)

#--------------------------------------


# aplico el modelo a los datos sin clase
dapply <- dataset[foto_mes == PARAM$input$future]

# aplico el modelo a los datos nuevos
prediccion <- predict(
  modelo,
  data.matrix(dapply[, campos_buenos, with = FALSE])
)

# genero la tabla de entrega
tb_entrega <- dapply[, list(numero_de_cliente, foto_mes)]
tb_entrega[, prob := prediccion]

# grabo las probabilidad del modelo
fwrite(tb_entrega,
  file = "prediccion.txt",
  sep = "\t"
)

# ordeno por probabilidad descendente
setorder(tb_entrega, -prob)


# genero archivos con los  "envios" mejores
# deben subirse "inteligentemente" a Kaggle para no malgastar submits
# si la palabra inteligentemente no le significa nada aun
# suba TODOS los archivos a Kaggle
# espera a la siguiente clase sincronica en donde el tema sera explicado

cortes <- seq(8000, 15000, by = 500)
for (envios in cortes) {
  tb_entrega[, Predicted := 0L]
  tb_entrega[1:envios, Predicted := 1L]

  fwrite(tb_entrega[, list(numero_de_cliente, Predicted)],
    file = paste0(PARAM$experimento, "_", envios, ".csv"),
    sep = ","
  )
}

cat("\n\nLa generacion de los archivos para Kaggle ha terminado\n")
