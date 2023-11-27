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
#INDICAR METODO DE IMPUTACION

PARAM$experimento <- "KA8240_ensamble_2_LRalto_neg_bagging_undsampl"

PARAM$input$dataset <- "./datasets/competencia_03.csv.gz"

# meses donde se entrena el modelo
PARAM$input$training <- c(201903,	201904,	201905,	201906,	201907,	201908,	201909, 201910,	201911,	201912,	
                          202001,	202002,	202003,	202004,	202005,	202006, 202007,	202008,	202009,	202010,	202011,	202012,	
                          202101,	202102,	202103,	202104, 202105, 202106, 202107)

PARAM$input$future <- c(202109) # meses donde se aplica el modelo

PARAM$finalmodel$semilla <- c(290497, 540187, 987851, 984497, 111893, 100103, 100189, 101987, 991981, 991987,
                              106853, 191071, 337511, 400067, 991751, 729191, 729199, 729203, 729217, 729257)

# hiperparametros optimos
PARAM$finalmodel$optim$num_iterations <- 105
PARAM$finalmodel$optim$learning_rate <- 0.706878458704326
PARAM$finalmodel$optim$feature_fraction <- 0.982006171936024
PARAM$finalmodel$optim$min_data_in_leaf <- 49889
PARAM$finalmodel$optim$num_leaves <- 13
PARAM$finalmodel$optim$neg_bagging_fraction <- 0.264617087916123

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui empieza el programa
setwd("~/buckets/b1")

# cargo el dataset donde voy a entrenar
dataset <- fread(PARAM$input$dataset, stringsAsFactors = TRUE)

# Feature Engineering Historico  ----------------------------------------------
campos_buenos <- setdiff(colnames(dataset), c('foto_mes','numero_de_cliente',"clase_ternaria", "clase01"))

# Catastrophe Analysis  -------------------------------------------------------
# zero ratio imputo nulos
dataset[foto_mes %in% c(202010, 202102), mcajeros_propios_descuentos := NA]
dataset[foto_mes %in% c(202010, 202102), ctarjeta_visa_descuentos := NA]
dataset[foto_mes %in% c(202010, 202102), mtarjeta_visa_descuentos := NA]
dataset[foto_mes %in% c(202010, 202102), ctarjeta_master_descuentos := NA]
dataset[foto_mes %in% c(202010, 202102), mtarjeta_master_descuentos := NA]
dataset[foto_mes == 202105, ccajas_depositos := NA]

dataset[ foto_mes==201901,  ctransferencias_recibidas  := NA ]
dataset[ foto_mes==201901,  mtransferencias_recibidas  := NA ]

dataset[ foto_mes==201902,  ctransferencias_recibidas  := NA ]
dataset[ foto_mes==201902,  mtransferencias_recibidas  := NA ]

dataset[ foto_mes==201903,  ctransferencias_recibidas  := NA ]
dataset[ foto_mes==201903,  mtransferencias_recibidas  := NA ]

dataset[ foto_mes==201904,  ctransferencias_recibidas  := NA ]
dataset[ foto_mes==201904,  mtransferencias_recibidas  := NA ]
dataset[ foto_mes==201904,  ctarjeta_visa_debitos_automaticos  :=  NA ]
dataset[ foto_mes==201904,  mttarjeta_visa_debitos_automaticos := NA ]
dataset[ foto_mes==201904,  Visa_mfinanciacion_limite := NA ]

dataset[ foto_mes==201905,  ctransferencias_recibidas  := NA ]
dataset[ foto_mes==201905,  mtransferencias_recibidas  := NA ]
dataset[ foto_mes==201905,  mrentabilidad     := NA ]
dataset[ foto_mes==201905,  mrentabilidad_annual     := NA ]
dataset[ foto_mes==201905,  mcomisiones      := NA ]
dataset[ foto_mes==201905,  mpasivos_margen  := NA ]
dataset[ foto_mes==201905,  mactivos_margen  := NA ]
dataset[ foto_mes==201905,  ctarjeta_visa_debitos_automaticos  := NA ]
dataset[ foto_mes==201905,  ccomisiones_otras := NA ]
dataset[ foto_mes==201905,  mcomisiones_otras := NA ]

dataset[ foto_mes==201910,  mpasivos_margen   := NA ]
dataset[ foto_mes==201910,  mactivos_margen   := NA ]
dataset[ foto_mes==201910,  ccomisiones_otras := NA ]
dataset[ foto_mes==201910,  mcomisiones_otras := NA ]
dataset[ foto_mes==201910,  mcomisiones       := NA ]
dataset[ foto_mes==201910,  mrentabilidad     := NA ]
dataset[ foto_mes==201910,  mrentabilidad_annual        := NA ]
dataset[ foto_mes==201910,  chomebanking_transacciones  := NA ]
dataset[ foto_mes==201910,  ctarjeta_visa_descuentos    := NA ]
dataset[ foto_mes==201910,  ctarjeta_master_descuentos  := NA ]
dataset[ foto_mes==201910,  mtarjeta_visa_descuentos    := NA ]
dataset[ foto_mes==201910,  mtarjeta_master_descuentos  := NA ]
dataset[ foto_mes==201910,  ccajeros_propios_descuentos := NA ]
dataset[ foto_mes==201910,  mcajeros_propios_descuentos := NA ]

dataset[ foto_mes==202001,  cliente_vip   := NA ]

dataset[ foto_mes==202006,  active_quarter   := NA ]
dataset[ foto_mes==202006,  internet   := NA ]
dataset[ foto_mes==202006,  mrentabilidad   := NA ]
dataset[ foto_mes==202006,  mrentabilidad_annual   := NA ]
dataset[ foto_mes==202006,  mcomisiones   := NA ]
dataset[ foto_mes==202006,  mactivos_margen   := NA ]
dataset[ foto_mes==202006,  mpasivos_margen   := NA ]
dataset[ foto_mes==202006,  mcuentas_saldo   := NA ]
dataset[ foto_mes==202006,  ctarjeta_debito_transacciones  := NA ]
dataset[ foto_mes==202006,  mautoservicio   := NA ]
dataset[ foto_mes==202006,  ctarjeta_visa_transacciones   := NA ]
dataset[ foto_mes==202006,  mtarjeta_visa_consumo   := NA ]
dataset[ foto_mes==202006,  ctarjeta_master_transacciones  := NA ]
dataset[ foto_mes==202006,  mtarjeta_master_consumo   := NA ]
dataset[ foto_mes==202006,  ccomisiones_otras   := NA ]
dataset[ foto_mes==202006,  mcomisiones_otras   := NA ]
dataset[ foto_mes==202006,  cextraccion_autoservicio   := NA ]
dataset[ foto_mes==202006,  mextraccion_autoservicio   := NA ]
dataset[ foto_mes==202006,  ccheques_depositados   := NA ]
dataset[ foto_mes==202006,  mcheques_depositados   := NA ]
dataset[ foto_mes==202006,  ccheques_emitidos   := NA ]
dataset[ foto_mes==202006,  mcheques_emitidos   := NA ]
dataset[ foto_mes==202006,  ccheques_depositados_rechazados   := NA ]
dataset[ foto_mes==202006,  mcheques_depositados_rechazados   := NA ]
dataset[ foto_mes==202006,  ccheques_emitidos_rechazados   := NA ]
dataset[ foto_mes==202006,  mcheques_emitidos_rechazados   := NA ]
dataset[ foto_mes==202006,  tcallcenter   := NA ]
dataset[ foto_mes==202006,  ccallcenter_transacciones   := NA ]
dataset[ foto_mes==202006,  thomebanking   := NA ]
dataset[ foto_mes==202006,  chomebanking_transacciones   := NA ]
dataset[ foto_mes==202006,  ccajas_transacciones   := NA ]
dataset[ foto_mes==202006,  ccajas_consultas   := NA ]
dataset[ foto_mes==202006,  ccajas_depositos   := NA ]
dataset[ foto_mes==202006,  ccajas_extracciones   := NA ]
dataset[ foto_mes==202006,  ccajas_otras   := NA ]
dataset[ foto_mes==202006,  catm_trx   := NA ]
dataset[ foto_mes==202006,  matm   := NA ]
dataset[ foto_mes==202006,  catm_trx_other   := NA ]
dataset[ foto_mes==202006,  matm_other   := NA ]
dataset[ foto_mes==202006,  ctrx_quarter   := NA ]
dataset[ foto_mes==202006,  cmobile_app_trx   := NA ]

#------------------------------------------------------------------------------
#Imputación de nulos

# Convierto integer a numeric (me permite poder hacer las cuentas del feature engeneering intrames)
for (i in colnames(dataset)) {
  if (class(dataset[[i]]) == "integer") {
    dataset[[i]] <- as.numeric(dataset[[i]])
  }
}

#------------------------------------------------------------------------------

# Data Drifting----------------------------------------------------------------
# por ahora, no hago nada

# Ranking de las variables expresadas en pesos pero sin centrar en cero
diccionario <- fread("~/buckets/b1/datasets/DiccionarioDatos_2023.csv")

features_pesos <- as.character(diccionario[unidad == 'pesos', campo])

# Loop through each column in features_pesos and calculate the rank
for (i in features_pesos) {
  dataset[, (i) := frankv(dataset[[i]], order = 1L, na.last = 'keep', ties.method = "dense")]
}

# Convierto integer a numeric (me permite poder hacer las cuentas del feature engeneering intrames)
for (i in colnames(dataset)) {
  if (class(dataset[[i]]) == "integer") {
    dataset[[i]] <- as.numeric(dataset[[i]])
  }
}

# Feature Engineering Intrames---------------------------------------------
dataset[, "rentabilidad_mensual" := mrentabilidad / fcoalesce(mrentabilidad_annual, 0)]

dataset[, "mcomisiones_rentabilidad" := mcomisiones / fcoalesce(mrentabilidad_annual, 0)]

dataset[, "m_activos_pasivos_margen_neto" := fcoalesce(mactivos_margen, 0) + fcoalesce(mpasivos_margen, 0)]

dataset[, "m_activos_pasivos_margen_absoluto" := abs(fcoalesce(mactivos_margen, 0)) + abs(fcoalesce(mpasivos_margen, 0))]

dataset[, "mtotal_cta_corriente" := fcoalesce(mcuenta_corriente_adicional, 0) + fcoalesce(mcuenta_corriente, 0)]

dataset[, "c_cuentas" := fcoalesce(ccuenta_corriente, 0) + fcoalesce(ccaja_ahorro, 0)]

dataset[, "mtotal_caja_ahorro" := fcoalesce(mcaja_ahorro, 0) + fcoalesce(mcaja_ahorro_adicional, 0) + fcoalesce(mcaja_ahorro_dolares, 0)]

dataset[, "porcentaje_saldo_cta_corriente" := (fcoalesce(mcuenta_corriente_adicional, 0) + fcoalesce(mcuenta_corriente, 0)) / fcoalesce(mcuentas_saldo, 0)]

dataset[, "porcentaje_saldo_caja_ahorro" := (fcoalesce(mcaja_ahorro, 0) + fcoalesce(mcaja_ahorro_adicional, 0) + fcoalesce(mcaja_ahorro_dolares, 0)) / fcoalesce(mcuentas_saldo, 0)]

dataset[, "cant_transacciones_por_tarjeta_debito" := ctarjeta_debito_transacciones / fcoalesce(ctarjeta_debito, 0)]

dataset[, "monto_prom_trx_tarjeta_debito" := ifelse(ctarjeta_debito != 0, mautoservicio * 1.0 / ctarjeta_debito, NA)]

dataset[, "cant_cuentas_visa_y_master" := fcoalesce(ctarjeta_visa, 0) + fcoalesce(ctarjeta_master, 0)]

dataset[, "cons_prom_visa" := ifelse(ctarjeta_visa_transacciones != 0, mtarjeta_master_consumo * 1.0 / ctarjeta_visa_transacciones, NA)]

dataset[, "cons_prom_master" := ifelse(ctarjeta_master_transacciones != 0, mtarjeta_master_consumo * 1.0 / ctarjeta_master_transacciones, NA)]

dataset[, "consumo_total_tc" := fcoalesce(mtarjeta_visa_consumo, 0) + fcoalesce(mtarjeta_master_consumo, 0)]

dataset[, "transacciones_totales_tc" := fcoalesce(ctarjeta_visa_transacciones, 0) + fcoalesce(ctarjeta_master_transacciones, 0)]

dataset[, "consumo_promedio_tc" := ifelse((ctarjeta_visa_transacciones + fcoalesce(ctarjeta_master_transacciones, 0)) != 0,
                                          (fcoalesce(mtarjeta_visa_consumo, 0) + fcoalesce(mtarjeta_master_consumo, 0)) * 1.0 / (ctarjeta_visa_transacciones + fcoalesce(ctarjeta_master_transacciones, 0)),
                                          NA)]

dataset[, "cantidad_total_prestamos" := fcoalesce(cprestamos_personales, 0) + fcoalesce(cprestamos_prendarios, 0) + fcoalesce(cprestamos_hipotecarios, 0)]

dataset[, "monto_total_prestamos" := fcoalesce(mprestamos_personales, 0) + fcoalesce(mprestamos_prendarios, 0) + fcoalesce(mprestamos_hipotecarios, 0)]

dataset[, "deuda_promedio_prestamo" := ifelse((cprestamos_personales + fcoalesce(cprestamos_prendarios, 0) + fcoalesce(cprestamos_hipotecarios, 0)) != 0,
                                              (fcoalesce(mprestamos_personales, 0) + fcoalesce(mprestamos_prendarios, 0) + fcoalesce(mprestamos_hipotecarios, 0)) / (cprestamos_personales + fcoalesce(cprestamos_prendarios, 0) + fcoalesce(cprestamos_hipotecarios, 0)),
                                              NA)]

dataset[, "monto_total_plazos_fijos" := fcoalesce(mplazo_fijo_dolares, 0) + fcoalesce(mplazo_fijo_pesos, 0)]

dataset[, "dinero_prom_por_plazo_fijo" := ifelse(cplazo_fijo != 0,
                                                 (fcoalesce(mplazo_fijo_dolares, 0) + fcoalesce(mplazo_fijo_pesos, 0)) / cplazo_fijo,
                                                 NA)]

dataset[, "monto_total_inv_tipo1" := fcoalesce(minversion1_pesos, 0) + fcoalesce(minversion1_dolares, 0)]


dataset[, "dinero_prom_inv_tipo1" := ifelse(cinversion1 != 0,
                                            (fcoalesce(minversion1_pesos, 0) + fcoalesce(minversion1_dolares, 0)) / cinversion1,
                                            NA)]


dataset[, "dinero_prom_inv_tipo2" := ifelse(cinversion1 != 0,
                                            fcoalesce(minversion2, 0) / cinversion1,
                                            NA)]

dataset[, "cant_inversiones_totales" := fcoalesce(cplazo_fijo, 0) + fcoalesce(cinversion1, 0) + fcoalesce(cinversion2, 0)]

dataset[, "monto_inversiones_totales" := fcoalesce(mplazo_fijo_dolares, 0) + fcoalesce(mplazo_fijo_pesos, 0) + fcoalesce(minversion1_pesos, 0) + fcoalesce(minversion1_dolares, 0) + fcoalesce(minversion2, 0)]

dataset[, "dinero_prom_por_inversion" := ifelse((cplazo_fijo + cinversion1 + cinversion2) != 0,
                                                (mplazo_fijo_dolares + mplazo_fijo_pesos + minversion1_pesos + minversion1_dolares + minversion2) /
                                                  (cplazo_fijo + cinversion1 + cinversion2),
                                                NA)]

dataset[, "cant_seguros_totales" := fcoalesce(cseguro_vida, 0) + fcoalesce(cseguro_auto, 0) + fcoalesce(cseguro_vivienda, 0) + fcoalesce(cseguro_accidentes_personales, 0)]

dataset[, "monto_total_acreditaciones" := fcoalesce(mpayroll, 0) + fcoalesce(mpayroll2, 0)]

dataset[, "cantidad_total_acreditaciones" := fcoalesce(cpayroll2_trx, 0) + fcoalesce(cpayroll_trx, 0)]

dataset[, "acreditaciones_sobre_edad" := (fcoalesce(mpayroll, 0) + fcoalesce(mpayroll2, 0)) / fcoalesce(cliente_edad, 0)]

dataset[, "monto_debito_promedio_cuenta" := ifelse(fcoalesce(ccuenta_debitos_automaticos, 0) != 0,
                                                   mcuenta_debitos_automaticos / fcoalesce(ccuenta_debitos_automaticos, 0),
                                                   NA)]

dataset[, "cantidad_debitos_en_tc" := fcoalesce(ctarjeta_visa_debitos_automaticos, 0) + fcoalesce(ctarjeta_master_debitos_automaticos, 0)]

dataset[, "monto_debitos_en_tc" := fcoalesce(mttarjeta_visa_debitos_automaticos, 0) + fcoalesce(mttarjeta_master_debitos_automaticos, 0)]

dataset[, "cantidad_debitos_totales" := fcoalesce(ccuenta_debitos_automaticos, 0) + fcoalesce(ctarjeta_visa_debitos_automaticos, 0) + fcoalesce(ctarjeta_master_debitos_automaticos, 0)]

dataset[, "monto_total_debitado_tc" := fcoalesce(mcuenta_debitos_automaticos, 0) + fcoalesce(mttarjeta_visa_debitos_automaticos, 0) + fcoalesce(mttarjeta_master_debitos_automaticos, 0)]

dataset[, "cantidad_pagos_servicio_totales" := fcoalesce(cpagodeservicios, 0) + fcoalesce(cpagomiscuentas, 0)]

dataset[, "monto_pago_servicios_totales" := fcoalesce(mpagodeservicios, 0) + fcoalesce(mpagomiscuentas, 0)]

dataset[, "descuento_cajero_promedio" := ifelse(fcoalesce(ccajeros_propios_descuentos, 0) != 0,
                                                mcajeros_propios_descuentos / fcoalesce(ccajeros_propios_descuentos, 0),
                                                NA)]

dataset[, "cantidad_descuentos_totales_tc" := fcoalesce(ctarjeta_visa_descuentos, 0) + fcoalesce(ctarjeta_master_descuentos, 0)]

dataset[, "cantidad_comisiones_totales" := fcoalesce(ccomisiones_mantenimiento, 0) + fcoalesce(ccomisiones_otras, 0)]

dataset[, "monto_total_comisiones" := fcoalesce(mcomisiones_mantenimiento, 0) + fcoalesce(mcomisiones_otras, 0)]

dataset[, "comision_promedio" := ifelse(fcoalesce((ccomisiones_mantenimiento + ccomisiones_otras), 0) != 0,
                                        (fcoalesce(mcomisiones_mantenimiento, 0) + fcoalesce(mcomisiones_otras, 0)) / fcoalesce((ccomisiones_mantenimiento + ccomisiones_otras), 0),
                                        NA)]


dataset[, "monto_total_operado_forex" := fcoalesce(mforex_buy, 0) + fcoalesce(mforex_sell, 0)]

# Calculate the column "monto_operado_promedio_forex" with handling division by zero
dataset[, "monto_operado_promedio_forex" := ifelse(fcoalesce(cforex, 0) != 0,
                                                   (fcoalesce(mforex_buy, 0) + fcoalesce(mforex_sell, 0)) / fcoalesce(cforex, 0),
                                                   NA)]


dataset[, "transferencias_totales" := fcoalesce(ctransferencias_recibidas, 0) + fcoalesce(ctransferencias_emitidas, 0)]

dataset[, "montos_transferencias_totales_absolutas" := abs(fcoalesce(mtransferencias_recibidas, 0)) + abs(fcoalesce(mtransferencias_emitidas, 0))]

dataset[, "montos_transferencias_totales_netas" := fcoalesce(mtransferencias_recibidas, 0) + fcoalesce(mtransferencias_emitidas, 0)]

dataset[, "monto_transferencia_promedio" := ifelse(fcoalesce((ctransferencias_recibidas + ctransferencias_emitidas), 0) != 0,
                                                   (abs(fcoalesce(mtransferencias_recibidas, 0)) + abs(fcoalesce(mtransferencias_emitidas, 0))) / fcoalesce((ctransferencias_recibidas + ctransferencias_emitidas), 0),
                                                   NA)]

dataset[, "extraccion_cajero_promedio" := ifelse(fcoalesce(cextraccion_autoservicio, 0) != 0,
                                                 mextraccion_autoservicio / fcoalesce(cextraccion_autoservicio, 0),
                                                 NA)]


dataset[, "cantidad_interacciones_con_linea_caja" := fcoalesce(ccajas_transacciones, 0) + fcoalesce(ccajas_consultas, 0) + fcoalesce(ccajas_depositos, 0) + fcoalesce(ccajas_extracciones, 0) + fcoalesce(ccajas_otras, 0)]

dataset[, "transacciones_totales_cajeros_automaticos" := fcoalesce(catm_trx, 0) + fcoalesce(catm_trx_other, 0)]

dataset[, "monto_total_cajeros_automaticos" := fcoalesce(matm, 0) + fcoalesce(matm_other, 0)]


dataset[, "monto_promedio_operado_cajero_automatico" := ifelse((fcoalesce(catm_trx, 0) + fcoalesce(catm_trx_other, 0)) != 0,
                                                               (fcoalesce(matm, 0) + fcoalesce(matm_other, 0)) / (fcoalesce(catm_trx, 0) + fcoalesce(catm_trx_other, 0)),
                                                               NA)]


dataset[, "master_consumo_total" := fcoalesce(Master_mconsumospesos, 0) + fcoalesce(Master_mconsumosdolares, 0)]

dataset[, "master_limite_compra_sobre_sueldo" := ifelse(fcoalesce((mpayroll + mpayroll2), 0) != 0,
                                                        Master_mlimitecompra / fcoalesce((mpayroll + mpayroll2), 0),
                                                        NA)]

dataset[, "extracciones_totales_master" := fcoalesce(Master_madelantopesos, 0) + fcoalesce(Master_madelantodolares, 0)]

dataset[, "visa_consumo_total" := fcoalesce(Visa_mconsumospesos, 0) + fcoalesce(Visa_mconsumosdolares, 0)]

dataset[, "visa_limite_compra_sobre_sueldo" := ifelse(fcoalesce((mpayroll + mpayroll2), 0) != 0,
                                                      Visa_mlimitecompra / fcoalesce((mpayroll + mpayroll2), 0),
                                                      NA)]

dataset[, "master_abierta" := (Master_status == 0)]

dataset[, "master_en_proceso_cierre" := (Master_status %in% c(6, 7))]

dataset[, "master_cerrada" := (Master_status == 9)]

dataset[, "visa_abierta" := (Visa_status == 0)]

dataset[, "visa_en_proceso_cierre" := (Visa_status %in% c(6, 7))]

dataset[, "visa_cerrada" := (Visa_status == 9)]

dataset[, "adelantos_dolares_tc" := fcoalesce(Master_madelantodolares, 0) + fcoalesce(Visa_madelantodolares, 0)]

dataset[, "mlimite_compra_total_tc" := fcoalesce(Master_mlimitecompra, 0) + fcoalesce(Visa_mlimitecompra, 0)]


# Feature Engineering Historico  ----------------------------------------------
campos_buenos <- setdiff(colnames(dataset), c('foto_mes','numero_de_cliente',"clase_ternaria", "clase01"))

#calculate lags 1,3 and 6 in campos_buenos
dataset[, paste0(campos_buenos, "_lag1") := lapply(.SD, shift, 1L, type = "lag"), .SDcols = campos_buenos]
dataset[, paste0(campos_buenos, "_lag3") := lapply(.SD, shift, 3L, type = "lag"), .SDcols = campos_buenos]
dataset[, paste0(campos_buenos, "_lag6") := lapply(.SD, shift, 6L, type = "lag"), .SDcols = campos_buenos]

# agrego los delta lags de orden 1
for (vcol in campos_buenos) dataset[, paste0(vcol, "_delta1") := get(vcol) - get(paste0(vcol, "_lag1"))]


# agrego los delta lags de orden 3
for (vcol in campos_buenos) dataset[, paste0(vcol, "_delta3") := get(vcol) - get(paste0(vcol, "_lag3"))]


# agrego los delta lags de orden 6
for (vcol in campos_buenos) dataset[, paste0(vcol, "_delta6") := get(vcol) - get(paste0(vcol, "_lag6"))]

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
tb_entrega <- dapply[, list(numero_de_cliente, foto_mes)]

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
    #neg_bagging_fraction = 1.0, # 0.0 < neg_bagging_fraction <= 1.0
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

#-------------------GENERO ENSEMBLE---------------------------------------------------#

# Calcular el promedio de las predicciones (probas) de las semillas ejecutadas (excluyo cols "numero_de_cliente" y "foto_mes")
tb_entrega$proba_ensemble <- rowMeans(tb_entrega[, .SD, .SDcols = -(1:2)])
cat("\n\nEnsemble generado")

# ordeno por probabilidad descendente
setorder(tb_entrega, -proba_ensemble)


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
