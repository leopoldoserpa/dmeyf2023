# Este script esta pensado para correr en Google Cloud
#   8 vCPU
# 128 GB memoria RAM

# se entrena con clase_binaria2  POS =  { BAJA+1, BAJA+2 }
# Optimizacion Bayesiana de hiperparametros de  lightgbm,

# limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

require("data.table")
require("rlist")

require("lightgbm")

# paquetes necesarios para la Bayesian Optimization
require("DiceKriging")
require("mlrMBO")

#Para imputacion
install.packages("mice")
library(mice)
library(dplyr)

# para que se detenga ante el primer error
# y muestre el stack de funciones invocadas
options(error = function() {
  traceback(20)
  options(error = NULL)
  stop("exiting after script error")
})



# defino los parametros de la corrida, en una lista, la variable global  PARAM
#  muy pronto esto se leera desde un archivo formato .yaml
PARAM <- list()

#INDICAR METODO DE IMPUTACION
metodo_imputacion <- "mice_lasso.norm_10pct_m1_maxit5"

PARAM$experimento <- paste("EC_HT8230",metodo_imputacion,sep="_")

PARAM$input$dataset <- "datasets/competencia_03.csv.gz"

# los meses en los que vamos a entrenar
#  mucha magia emerger de esta eleccion
PARAM$input$testing <- c(202105)
PARAM$input$validation <- c(202104)
PARAM$input$training <- c(202010,202011,202012, 202101, 202102, 202103)

# un undersampling de 0.1  toma solo el 10% de los CONTINUA
PARAM$trainingstrategy$undersampling <- 1.0
PARAM$trainingstrategy$semilla_azar <- 290497 # Aqui poner su  primer  semilla

PARAM$hyperparametertuning$POS_ganancia <- 273000
PARAM$hyperparametertuning$NEG_ganancia <- -7000

# Aqui poner su segunda semilla
PARAM$lgb_semilla <- 540187
#------------------------------------------------------------------------------

# Hiperparametros FIJOS de  lightgbm
PARAM$lgb_basicos <- list(
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
  num_iterations = 9999, # un numero muy grande, lo limita early_stopping_rounds

  bagging_fraction = 1.0, # 0.0 < bagging_fraction <= 1.0
  pos_bagging_fraction = 1.0, # 0.0 < pos_bagging_fraction <= 1.0
  neg_bagging_fraction = 1.0, # 0.0 < neg_bagging_fraction <= 1.0
  is_unbalance = FALSE, #
  scale_pos_weight = 1.0, # scale_pos_weight > 0.0

  drop_rate = 0.1, # 0.0 < neg_bagging_fraction <= 1.0
  max_drop = 50, # <=0 means no limit
  skip_drop = 0.5, # 0.0 <= skip_drop <= 1.0

  extra_trees = TRUE, # Magic Sauce

  seed = PARAM$lgb_semilla
)


# Aqui se cargan los hiperparametros que se optimizan
#  en la Bayesian Optimization
PARAM$bo_lgb <- makeParamSet(
  makeNumericParam("learning_rate", lower = 0.02, upper = 0.3),
  makeNumericParam("feature_fraction", lower = 0.01, upper = 1.0),
  makeIntegerParam("num_leaves", lower = 8L, upper = 1024L),
  makeIntegerParam("min_data_in_leaf", lower = 100L, upper = 50000L)
)

# si usted es ambicioso, y tiene paciencia, podria subir este valor a 100
PARAM$bo_iteraciones <- 50 # iteraciones de la Optimizacion Bayesiana

#------------------------------------------------------------------------------
# graba a un archivo los componentes de lista
# para el primer registro, escribe antes los titulos

loguear <- function(
    reg, arch = NA, folder = "./exp/",
    ext = ".txt", verbose = TRUE) {
  archivo <- arch
  if (is.na(arch)) archivo <- paste0(folder, substitute(reg), ext)

  if (!file.exists(archivo)) # Escribo los titulos
    {
      linea <- paste0(
        "fecha\t",
        paste(list.names(reg), collapse = "\t"), "\n"
      )

      cat(linea, file = archivo)
    }

  linea <- paste0(
    format(Sys.time(), "%Y%m%d %H%M%S"), "\t", # la fecha y hora
    gsub(", ", "\t", toString(reg)), "\n"
  )

  cat(linea, file = archivo, append = TRUE) # grabo al archivo

  if (verbose) cat(linea) # imprimo por pantalla
}
#------------------------------------------------------------------------------
GLOBAL_arbol <- 0L
GLOBAL_gan_max <- -Inf
vcant_optima <- c()

fganancia_lgbm_meseta <- function(probs, datos) {
  vlabels <- get_field(datos, "label")
  vpesos <- get_field(datos, "weight")


  GLOBAL_arbol <<- GLOBAL_arbol + 1
  tbl <- as.data.table(list(
    "prob" = probs,
    "gan" = ifelse(vlabels == 1 & vpesos > 1,
      PARAM$hyperparametertuning$POS_ganancia,
      PARAM$hyperparametertuning$NEG_ganancia  )
  ))

  setorder(tbl, -prob)
  tbl[, posicion := .I]
  tbl[, gan_acum := cumsum(gan)]

  tbl[, gan_suavizada :=
    frollmean(
      x = gan_acum, n = 2001, align = "center",
      na.rm = TRUE, hasNA = TRUE
    )]

  gan <- tbl[, max(gan_suavizada, na.rm = TRUE)]


  pos <- which.max(tbl[, gan_suavizada])
  vcant_optima <<- c(vcant_optima, pos)

  if (GLOBAL_arbol %% 10 == 0) {
    if (gan > GLOBAL_gan_max) GLOBAL_gan_max <<- gan

    cat("\r")
    cat(
      "Validate ", GLOBAL_iteracion, " ", " ",
      GLOBAL_arbol, "  ", gan, "   ", GLOBAL_gan_max, "   "
    )
  }


  return(list(
    "name" = "ganancia",
    "value" = gan,
    "higher_better" = TRUE
  ))
}
#------------------------------------------------------------------------------

EstimarGanancia_lightgbm <- function(x) {
  gc()
  GLOBAL_iteracion <<- GLOBAL_iteracion + 1L

  # hago la union de los parametros basicos y los moviles que vienen en x
  param_completo <- c(PARAM$lgb_basicos, x)

  param_completo$early_stopping_rounds <-
    as.integer(400 + 4 / param_completo$learning_rate)

  GLOBAL_arbol <<- 0L
  GLOBAL_gan_max <<- -Inf
  vcant_optima <<- c()
  set.seed(PARAM$lgb_semilla, kind = "L'Ecuyer-CMRG")
  modelo_train <- lgb.train(
    data = dtrain,
    valids = list(valid = dvalidate),
    eval = fganancia_lgbm_meseta,
    param = param_completo,
    verbose = -100
  )

  cat("\n")

  cant_corte <- vcant_optima[modelo_train$best_iter]

  # aplico el modelo a testing y calculo la ganancia
  prediccion <- predict(
    modelo_train,
    data.matrix(dataset_test[, campos_buenos, with = FALSE])
  )

  tbl <- copy(dataset_test[, list("gan" = ifelse(clase_ternaria == "BAJA+2",
    PARAM$hyperparametertuning$POS_ganancia, 
    PARAM$hyperparametertuning$NEG_ganancia))])

  tbl[, prob := prediccion]
  setorder(tbl, -prob)
  tbl[, gan_acum := cumsum(gan)]
  tbl[, gan_suavizada := frollmean(
    x = gan_acum, n = 2001,
    align = "center", na.rm = TRUE, hasNA = TRUE
  )]


  ganancia_test <- tbl[, max(gan_suavizada, na.rm = TRUE)]

  cantidad_test_normalizada <- which.max(tbl[, gan_suavizada])

  rm(tbl)
  gc()

  ganancia_test_normalizada <- ganancia_test


  # voy grabando las mejores column importance
  if (ganancia_test_normalizada > GLOBAL_gananciamax) {
    GLOBAL_gananciamax <<- ganancia_test_normalizada
    tb_importancia <- as.data.table(lgb.importance(modelo_train))

    fwrite(tb_importancia,
      file = paste0("impo_", sprintf("%03d", GLOBAL_iteracion), ".txt"),
      sep = "\t"
    )

    rm(tb_importancia)
  }


  # logueo final
  ds <- list("cols" = ncol(dtrain), "rows" = nrow(dtrain))
  xx <- c(ds, copy(param_completo))

  xx$early_stopping_rounds <- NULL
  xx$num_iterations <- modelo_train$best_iter
  xx$estimulos <- cantidad_test_normalizada
  xx$ganancia <- ganancia_test_normalizada
  xx$iteracion_bayesiana <- GLOBAL_iteracion

  loguear(xx, arch = "BO_log.txt")

  set.seed(PARAM$lgb_semilla, kind = "L'Ecuyer-CMRG")
  return(ganancia_test_normalizada)
}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui empieza el programa

# Aqui se debe poner la carpeta de la computadora local
setwd("~/buckets/b1/") # Establezco el Working Directory

# cargo el dataset donde voy a entrenar el modelo
dataset <- fread(PARAM$input$dataset)


# creo la carpeta donde va el experimento
dir.create("./exp/", showWarnings = FALSE)
dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)

# Establezco el Working Directory DEL EXPERIMENTO
setwd(paste0("./exp/", PARAM$experimento, "/"))

# en estos archivos quedan los resultados
kbayesiana <- paste0(PARAM$experimento, ".RDATA")
klog <- paste0(PARAM$experimento, ".txt")

# Feature Engineering Historico  ----------------------------------------------
campos_buenos <- setdiff(colnames(dataset), c('foto_mes','numero_de_cliente',"clase_ternaria", "clase01"))


# zero ratio imputo nulos

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

#10% de los continuan en el periodo de entrenamiento
continua_training_0.1 <- sample_frac(dataset[dataset$foto_mes %in% c(202010,202011,202012, 202101, 202102, 202103)
                                             & clase_ternaria == "CONTINUA"], 0.1, seed = PARAM$lgb_semilla)

#100% BAJA+1 Y BAJA+2
no_continua_training <- dataset[dataset$foto_mes %in% c(202010,202011,202012, 202101, 202102, 202103) 
                                & clase_ternaria != "CONTINUA"]

training_subsampling_continua <- rbind(continua_training_0.1, no_continua_training)

time_imp.train <- system.time({imp.train <- mice(
  data = training_subsampling_continua[,..campos_buenos], 
  method = 'lasso.norm', 
  seed = PARAM$lgb_semilla, 
  #n.core = 3, 
  m = 1,
  maxit = 5,
  printFlag = TRUE
  #verbose = TRUE,
  #ignore = ignored
)})


time_imp.test <-  system.time({imp.test <- mice.mids(imp.train, newdata = dataset[foto_mes %in% c(202105),..campos_buenos],
                                                  maxit=5,printFlag = T)})

time_imp.val <-system.time({imp.val <- mice.mids(imp.train, newdata = dataset[foto_mes %in% c(202104),..campos_buenos],
                                  maxit=5,printFlag = T)})

time_imp.train_full <- system.time({imp.train_full <- mice.mids(imp.train,newdata = dataset[foto_mes %in% c(202010,202011,202012, 202101, 202102, 202103),..campos_buenos], 
                                               maxit=5,printFlag = T)})

# Chequeo nulos
sum(is.na(dataset[foto_mes %in% c(202010, 202011, 202012, 202101, 202102, 202103, 202104, 202105)]))

# Reemplazo 
dataset[foto_mes %in% c(202010,202011,202012, 202101, 202102, 202103),campos_buenos] <- complete(imp.train_full)
dataset[foto_mes %in% c(202104),campos_buenos] <- complete(imp.val)
dataset[foto_mes %in% c(202105),campos_buenos] <- complete(imp.test)

# Chequeo que no hay nulos
sum(is.na(dataset[foto_mes %in% c(202010, 202011, 202012, 202101, 202102, 202103, 202104, 202105)]))

------------------#Extraigo tiempo de ejecucion
# Extract relevant information
time_imp.train <- data.frame(
  user = time_imp.train[1],
  system = time_imp.train[2],
  elapsed = time_imp.train[3]
)

time_imp.test <- data.frame(
  user = time_imp.test[1],
  system = time_imp.test[2],
  elapsed = time_imp.test[3]
)

time_imp.val <- data.frame(
  user = time_imp.val[1],
  system = time_imp.val[2],
  elapsed = time_imp.val[3]
)

time_imp.train_full <- data.frame(
  user = time_imp.train_full[1],
  system = time_imp.train_full[2],
  elapsed = time_imp.train_full[3]
)

# Create a new column with the desired row names
time_imp.train$dataset <- "train"
time_imp.test$dataset <- "test"
time_imp.val$dataset <- "val"
time_imp.train_full$dataset <- "train_full"

# Combine the data frames
combined_data <- rbind(time_imp.train, time_imp.test, time_imp.val, time_imp.train_full)

# Set the row names based on the "dataset" column
rownames(combined_data) <- combined_data$dataset

# Remove the "dataset" column if not needed
combined_data <- combined_data[, -ncol(combined_data)]

# Write to CSV
write.csv(combined_data, file = "~/buckets/b1/PARAM$experimento/time_results.csv")

#------------------------------------------------------------------------------

# LAGS 1, 3 y 6 ################################################################

#calculate lags 1,3 and 6 in campos_buenos
dataset[, paste0(campos_buenos, "_lag1") := lapply(.SD, shift, 1L, type = "lag"), .SDcols = campos_buenos]
dataset[, paste0(campos_buenos, "_lag3") := lapply(.SD, shift, 3L, type = "lag"), .SDcols = campos_buenos]
dataset[, paste0(campos_buenos, "_lag6") := lapply(.SD, shift, 6L, type = "lag"), .SDcols = campos_buenos]

################################################################################



# ahora SI comienza la optimizacion Bayesiana

GLOBAL_iteracion <- 0 # inicializo la variable global
GLOBAL_gananciamax <- -1 # inicializo la variable global

# si ya existe el archivo log, traigo hasta donde llegue
if (file.exists(klog)) {
  tabla_log <- fread(klog)
  GLOBAL_iteracion <- nrow(tabla_log)
  GLOBAL_gananciamax <- tabla_log[, max(ganancia)]
}



# paso la clase a binaria que tome valores {0,1}  enteros
dataset[, clase01 := ifelse(clase_ternaria == "CONTINUA", 0L, 1L)]


# los campos que se van a utilizar
campos_buenos <- setdiff(
  colnames(dataset),
  c("clase_ternaria", "clase01", "azar", "training")
)

# defino los datos que forma parte del training
# aqui se hace el undersampling de los CONTINUA
set.seed(PARAM$trainingstrategy$semilla_azar)
dataset[, azar := runif(nrow(dataset))]
dataset[, training := 0L]
dataset[
  foto_mes %in% PARAM$input$training &
    (azar <= PARAM$trainingstrategy$undersampling | clase_ternaria %in% c("BAJA+1", "BAJA+2")),
  training := 1L
]

# dejo los datos en el formato que necesita LightGBM
dtrain <- lgb.Dataset(
  data = data.matrix(dataset[training == 1L, campos_buenos, with = FALSE]),
  label = dataset[training == 1L, clase01],
  weight = dataset[training == 1L, 
    ifelse(clase_ternaria == "BAJA+2", 1.0000001, 
      ifelse(clase_ternaria == "BAJA+1", 1.0, 1.0))],
  free_raw_data = FALSE
)



# defino los datos que forman parte de validation
#  no hay undersampling
dataset[, validation := 0L]
dataset[ foto_mes %in% PARAM$input$validation,  validation := 1L]

dvalidate <- lgb.Dataset(
  data = data.matrix(dataset[validation == 1L, campos_buenos, with = FALSE]),
  label = dataset[validation == 1L, clase01],
  weight = dataset[validation == 1L, 
    ifelse(clase_ternaria == "BAJA+2", 1.0000001, 
      ifelse(clase_ternaria == "BAJA+1", 1.0, 1.0))],
  free_raw_data = FALSE
)


# defino los datos de testing
dataset[, testing := 0L]
dataset[ foto_mes %in% PARAM$input$testing,  testing := 1L]


dataset_test <- dataset[testing == 1, ]

# libero espacio
rm(dataset)
gc()

# Aqui comienza la configuracion de la Bayesian Optimization
funcion_optimizar <- EstimarGanancia_lightgbm # la funcion que voy a maximizar

configureMlr(show.learner.output = FALSE)

# configuro la busqueda bayesiana,  los hiperparametros que se van a optimizar
# por favor, no desesperarse por lo complejo
obj.fun <- makeSingleObjectiveFunction(
  fn = funcion_optimizar, # la funcion que voy a maximizar
  minimize = FALSE, # estoy Maximizando la ganancia
  noisy = TRUE,
  par.set = PARAM$bo_lgb, # definido al comienzo del programa
  has.simple.signature = FALSE # paso los parametros en una lista
)

# cada 600 segundos guardo el resultado intermedio
ctrl <- makeMBOControl(
  save.on.disk.at.time = 600, # se graba cada 600 segundos
  save.file.path = kbayesiana
) # se graba cada 600 segundos

# indico la cantidad de iteraciones que va a tener la Bayesian Optimization
ctrl <- setMBOControlTermination(
  ctrl,
  iters = PARAM$bo_iteraciones
) # cantidad de iteraciones

# defino el método estandar para la creacion de los puntos iniciales,
# los "No Inteligentes"
ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())


# establezco la funcion que busca el maximo
surr.km <- makeLearner(
  "regr.km",
  predict.type = "se",
  covtype = "matern3_2",
  control = list(trace = TRUE)
)

# inicio la optimizacion bayesiana
if (!file.exists(kbayesiana)) {
  run <- mbo(obj.fun, learner = surr.km, control = ctrl)
} else {
  run <- mboContinue(kbayesiana) # retomo en caso que ya exista
}


cat("\n\nLa optimizacion Bayesiana ha terminado\n")
