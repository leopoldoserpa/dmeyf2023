# Optimizacion Bayesiana de hiperparametros de  rpart
# 10-repeated 5-fold Cross Validation
# trabaja con la clase_binaria2
#   POS={BAJA+1, BAJA+2}   NEG={CONTINUA}
# prueba el nivel optimo de OVERSAMPLING

# limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

require("data.table")
require("rlist")

require("rpart")
require("parallel")

# paquetes necesarios para la Bayesian Optimization
require("DiceKriging")
require("mlrMBO")


# Defino la  Optimizacion Bayesiana
PARAM <- list()

PARAM$home <- "~/buckets/b1/"

PARAM$experimento <- "HT3551"

# Aqui van las 10 semillas que hacen el 10-repeated
#  si se pone una sola semilla, se esta haciendo solo 5-fold xval
PARAM$semilla_azar <- c(
  100103, 100189, 101987, 991981, 991987, #mis 5 semillas
  753587, 247759, 253369, 955127, 800519
)

# folds del cross validation
PARAM$xval_folds <- 5

# cantidad de iteraciones de la Optimizacion Bayesiana
PARAM$BO_iter <- 100

# la letra L al final de 1L significa ENTERO
PARAM$hs <- makeParamSet(
  makeNumericParam("cp", lower = -1L, upper = -1L),
  makeIntegerParam("minsplit", lower = 500L, upper = 1500L),
  makeIntegerParam("minbucket", lower = 200L, upper = 800L),
  makeIntegerParam("maxdepth", lower = 6L, upper = 12L),
  makeIntegerParam("corte", lower = 9500L, upper = 9500L),
  makeIntegerParam("oversampling", lower = 100L, upper = 100L),
  forbidden = quote(minbucket > 0.5 * minsplit)
)
# minbuket NO PUEDE ser mayor que la mitad de minsplit


# este valor debe ser 1 si se utiliza Windows
PARAM$cores <- 1

#------------------------------------------------------------------------------
# graba a un archivo los componentes de lista
# para el primer registro, escribe antes los titulos

loguear <- function(reg, arch = NA, folder = "./work/", ext = ".txt",
                    verbose = TRUE) {
  archivo <- arch
  if (is.na(arch)) archivo <- paste0(folder, substitute(reg), ext)

  # Escribo los titulos
  if (!file.exists(archivo)) {
    linea <- paste0(
      "fecha\t",
      paste(list.names(reg), collapse = "\t"), "\n"
    )

    cat(linea, file = archivo)
  }

  # la fecha y hora
  linea <- paste0(
    format(Sys.time(), "%Y%m%d %H%M%S"), "\t",
    gsub(", ", "\t", toString(reg)), "\n"
  )

  # grabo al archivo
  cat(linea, file = archivo, append = TRUE)

  # imprimo por pantalla
  if (verbose) cat(linea)
}
#------------------------------------------------------------------------------
# particionar agrega una columna llamada fold a un dataset
#   que consiste en una particion estratificada segun agrupa
# particionar( data=dataset, division=c(70,30),
#  agrupa=clase_ternaria, seed=semilla)   crea una particion 70, 30
# particionar( data=dataset, division=c(1,1,1,1,1),
#  agrupa=clase_ternaria, seed=semilla)   divide el dataset en 5 particiones

particionar <- function(data, division, agrupa = "", campo = "fold",
                        start = 1, seed = NA) {
  if (!is.na(seed)) set.seed(seed)

  bloque <- unlist(mapply(
    function(x, y) {
      rep(y, x)
    }, division,
    seq(from = start, length.out = length(division))
  ))

  data[, (campo) := sample(rep(bloque, ceiling(.N / length(bloque))))[1:.N],
    by = agrupa
  ]
}
#------------------------------------------------------------------------------
# fold_test  tiene el numero de fold que voy a usar para testear,
#  entreno en el resto de los folds
# param tiene los hiperparametros del arbol

ArbolSimple <- function(fold_test, data, param, qfolds) {
  # genero el modelo
  # entreno en todo MENOS el fold_test que uso para testing

  vpesos <<- rep(1, data[fold != fold_test, .N])

  if (param$oversampling > 0) {
    vpesos <<- data[
      fold != fold_test,
      ifelse(clase_ternaria == "CONTINUA", 1, param$oversampling)
    ]
  }

  modelo <- rpart("clase_virtual ~ . - clase_ternaria",
    data = data[fold != fold_test, ],
    xval = 0,
    control = param,
    weights = vpesos
  )

  # aplico el modelo a los datos de testing
  # aplico el modelo sobre los datos de testing
  # quiero que me devuelva probabilidades
  prediccion <- predict(modelo,
    data[fold == fold_test, ],
    type = "prob"
  )

  # esta es la probabilidad de baja
  prob_pos <- prediccion[, "POS"]

  tb_final <- copy(data[fold == fold_test, list(clase_ternaria)])
  tb_final[, prob := prob_pos]
  setorder(tb_final, -prob)

  # calculo la ganancia, los PARAM$corte de mayor probabilidad
  ganancia_testing <- tb_final[
    1:round(param$corte / qfolds),
    sum(ifelse(clase_ternaria == "BAJA+2",
      273000, -7000
    ))
  ]

  rm(tb_final)
  # esta es la ganancia sobre el fold de testing, NO esta normalizada
  return(ganancia_testing)
}
#------------------------------------------------------------------------------

ArbolesCrossValidation <- function(data, param, qfolds, pagrupa, semilla) {
  # generalmente  c(1, 1, 1, 1, 1 )  cinco unos
  divi <- rep(1, qfolds)

  # particiono en dataset en folds
  particionar(data, divi, seed = semilla, agrupa = pagrupa)

  ganancias <- mcmapply(ArbolSimple,
    seq(qfolds), # 1 2 3 4 5
    MoreArgs = list(data, param, qfolds),
    SIMPLIFY = FALSE,
    mc.cores = PARAM$cores
  )

  data[, fold := NULL]

  ganancia_xval <- sum(unlist(ganancias))

  return(ganancia_xval)
}
#------------------------------------------------------------------------------
# esta funcion solo puede recibir los parametros que se estan optimizando
# el resto de los parametros, lamentablemente se pasan como variables globales

EstimarGanancia <- function(x) {
  GLOBAL_iteracion <<- GLOBAL_iteracion + 1

  # param= x los hiperparametros del arbol
  # qfolds= PARAM$xval_folds  la cantidad de folds
  gans <- c()
  for (semilla in PARAM$semilla_azar)
  {
    ganancia <- ArbolesCrossValidation(dataset,
      param = x,
      qfolds = PARAM$xval_folds,
      pagrupa = "clase_ternaria",
      semilla = semilla
    )

    gans <- c(gans, ganancia)
  }

  cat(gans, "\n")
  # logueo
  xx <- x

  xx$xval_repeats <- length(PARAM$semilla_azar)
  xx$xval_folds <- PARAM$xval_folds

  xx$ganancia <- mean(gans)
  xx$iteracion <- GLOBAL_iteracion
  loguear(xx, arch = archivo_log)

  return(mean(gans))
}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui empieza el programa

# Establezco el Working Directory inicial
setwd("C:/Users/leopo/Desktop/Maestria DATA MINING/DMEyF")

# cargo los datos
dataset <- fread("./datasets/competencia_01.csv")

#dataset[, if_max_POS := (ctrx_quarter < 60) & (mpayroll < 7110.6) & (mpasivos_margen < 122.3)]

# entreno en 202103
dataset <- dataset[foto_mes == 202103]

dataset[, clase_virtual := ifelse(clase_ternaria == "CONTINUA", "NEG", "POS")]

# creo la carpeta donde va el experimento
#  HT  representa  Hiperparameter Tuning
dir.create("./exp/", showWarnings = FALSE)

dir.create(paste0("./exp/", PARAM$experimento, "/"),
  showWarnings = FALSE
)

# Establezco el Working Directory DEL EXPERIMENTO
setwd(paste0("./exp/", PARAM$experimento, "/"))


archivo_log <- "BO_log.txt"
archivo_BO <- "bayesiana.RDATA"

# leo si ya existe el log
#  para retomar en caso que se se corte el programa
GLOBAL_iteracion <- 0

if (file.exists(archivo_log)) {
  tabla_log <- fread(archivo_log)
  GLOBAL_iteracion <- nrow(tabla_log)
}



# Aqui comienza la configuracion de la Bayesian Optimization

funcion_optimizar <- EstimarGanancia

configureMlr(show.learner.output = FALSE)

# configuro la busqueda bayesiana,
#  los hiperparametros que se van a optimizar
# por favor, no desesperarse por lo complejo
# minimize= FALSE estoy Maximizando la ganancia
obj.fun <- makeSingleObjectiveFunction(
  fn = funcion_optimizar,
  minimize = FALSE,
  noisy = TRUE,
  par.set = PARAM$hs,
  has.simple.signature = FALSE
)

ctrl <- makeMBOControl(
  save.on.disk.at.time = 600,
  save.file.path = archivo_BO
)

ctrl <- setMBOControlTermination(ctrl, iters = PARAM$BO_iter)
ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

surr.km <- makeLearner("regr.km",
  predict.type = "se",
  covtype = "matern3_2", control = list(trace = TRUE)
)

# inicio la optimizacion bayesiana
if (!file.exists(archivo_BO)) {
  run <- mbo(
    fun = obj.fun,
    learner = surr.km,
    control = ctrl
  )
} else {
  run <- mboContinue(archivo_BO)
}
# retomo en caso que ya exista
