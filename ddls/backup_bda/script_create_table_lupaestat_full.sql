CREATE TABLE codigo_craai_mun_comarca(  municipio string,   craai string,   cod_municipio int,   cod_craai int,   cod_comarca int,   comarca string)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/lupaestat.db/codigo_craai_mun_comarca') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/lupaestat.db/codigo_craai_mun_comarca'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"municipio\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"craai\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cod_municipio\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cod_craai\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"cod_comarca\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"comarca\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='4958',   'transient_lastDdlTime'='1567451695')
CREATE TABLE escolas(  nu_ano_censo int,   co_entidade int,   no_entidade string,   co_orgao_regional string,   tp_situacao_funcionamento int,   dt_ano_letivo_inicio string,   dt_ano_letivo_termino string,   co_regiao int,   co_mesorregiao int,   co_microrregiao int,   co_uf int,   co_municipio int,   co_distrito int,   tp_dependencia int,   tp_localizacao int,   tp_categoria_escola_privada int,   in_conveniada_pp int,   tp_convenio_poder_publico int,   in_mant_escola_privada_emp int,   in_mant_escola_privada_ong int,   in_mant_escola_privada_sind int,   in_mant_escola_privada_sist_s int,   in_mant_escola_privada_s_fins int,   co_escola_sede_vinculada int,   co_ies_ofertante int,   tp_regulamentacao int,   in_local_func_predio_escolar int,   tp_ocupacao_predio_escolar int,   in_local_func_salas_empresa int,   in_local_func_socioeducativo int,   in_local_func_unid_prisional int,   in_local_func_prisional_socio int,   in_local_func_templo_igreja int,   in_local_func_casa_professor int,   in_local_func_galpao int,   tp_ocupacao_galpao int,   in_local_func_salas_outra_esc int,   in_local_func_outros int,   in_predio_compartilhado int,   in_agua_filtrada int,   in_agua_rede_publica int,   in_agua_poco_artesiano int,   in_agua_cacimba int,   in_agua_fonte_rio int,   in_agua_inexistente int,   in_energia_rede_publica int,   in_energia_gerador int,   in_energia_outros int,   in_energia_inexistente int,   in_esgoto_rede_publica int,   in_esgoto_fossa int,   in_esgoto_inexistente int,   in_lixo_coleta_periodica int,   in_lixo_queima int,   in_lixo_joga_outra_area int,   in_lixo_recicla int,   in_lixo_enterra int,   in_lixo_outros int,   in_sala_diretoria int,   in_sala_professor int,   in_laboratorio_informatica int,   in_laboratorio_ciencias int,   in_sala_atendimento_especial int,   in_quadra_esportes_coberta int,   in_quadra_esportes_descoberta int,   in_quadra_esportes int,   in_cozinha int,   in_biblioteca int,   in_sala_leitura int,   in_biblioteca_sala_leitura int,   in_parque_infantil int,   in_bercario int,   in_banheiro_fora_predio int,   in_banheiro_dentro_predio int,   in_banheiro_ei int,   in_banheiro_pne int,   in_dependencias_pne int,   in_secretaria int,   in_banheiro_chuveiro int,   in_refeitorio int,   in_despensa int,   in_almoxarifado int,   in_auditorio int,   in_patio_coberto int,   in_patio_descoberto int,   in_alojam_aluno int,   in_alojam_professor int,   in_area_verde int,   in_lavanderia int,   in_dependencias_outras int,   qt_salas_existentes int,   qt_salas_utilizadas int,   in_equip_tv int,   in_equip_videocassete int,   in_equip_dvd int,   in_equip_parabolica int,   in_equip_copiadora int,   in_equip_retroprojetor int,   in_equip_impressora int,   in_equip_impressora_mult int,   in_equip_som int,   in_equip_multimidia int,   in_equip_fax int,   in_equip_foto int,   in_computador int,   qt_equip_tv int,   qt_equip_videocassete int,   qt_equip_dvd int,   qt_equip_parabolica int,   qt_equip_copiadora int,   qt_equip_retroprojetor int,   qt_equip_impressora int,   qt_equip_impressora_mult int,   qt_equip_som int,   qt_equip_multimidia int,   qt_equip_fax int,   qt_equip_foto int,   qt_computador int,   qt_comp_administrativo int,   qt_comp_aluno int,   in_internet int,   in_banda_larga int,   qt_funcionarios int,   in_alimentacao int,   tp_aee int,   tp_atividade_complementar int,   in_fundamental_ciclos int,   tp_localizacao_diferenciada int,   in_material_esp_quilombola int,   in_material_esp_indigena int,   in_material_esp_nao_utiliza int,   in_educacao_indigena int,   tp_indigena_lingua int,   co_lingua_indigena int,   in_brasil_alfabetizado int,   in_final_semana int,   in_formacao_alternancia int,   in_mediacao_presencial int,   in_mediacao_semipresencial int,   in_mediacao_ead int,   in_especial_exclusiva int,   in_regular int,   in_eja int,   in_profissionalizante int,   in_comum_creche int,   in_comum_pre int,   in_comum_fund_ai int,   in_comum_fund_af int,   in_comum_medio_medio int,   in_comum_medio_integrado int,   in_comum_medio_normal int,   in_esp_exclusiva_creche int,   in_esp_exclusiva_pre int,   in_esp_exclusiva_fund_ai int,   in_esp_exclusiva_fund_af int,   in_esp_exclusiva_medio_medio int,   in_esp_exclusiva_medio_integr int,   in_esp_exclusiva_medio_normal int,   in_comum_eja_fund int,   in_comum_eja_medio int,   in_comum_eja_prof int,   in_esp_exclusiva_eja_fund int,   in_esp_exclusiva_eja_medio int,   in_esp_exclusiva_eja_prof int,   in_comum_prof int,   in_esp_exclusiva_prof int)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/lupaestat.db/escolas') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/lupaestat.db/escolas'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='2',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='4',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"NU_ANO_CENSO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_ENTIDADE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"NO_ENTIDADE\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_ORGAO_REGIONAL\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_SITUACAO_FUNCIONAMENTO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"DT_ANO_LETIVO_INICIO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"DT_ANO_LETIVO_TERMINO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_REGIAO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_MESORREGIAO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_MICRORREGIAO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_UF\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_MUNICIPIO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_DISTRITO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_DEPENDENCIA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_LOCALIZACAO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_CATEGORIA_ESCOLA_PRIVADA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_CONVENIADA_PP\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_CONVENIO_PODER_PUBLICO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_MANT_ESCOLA_PRIVADA_EMP\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_MANT_ESCOLA_PRIVADA_ONG\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_MANT_ESCOLA_PRIVADA_SIND\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_MANT_ESCOLA_PRIVADA_SIST_S\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_MANT_ESCOLA_PRIVADA_S_FINS\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_ESCOLA_SEDE_VINCULADA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_IES_OFERTANTE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_REGULAMENTACAO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LOCAL_FUNC_PREDIO_ESCOLAR\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_OCUPACAO_PREDIO_ESCOLAR\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LOCAL_FUNC_SALAS_EMPRESA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LOCAL_FUNC_SOCIOEDUCATIVO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LOCAL_FUNC_UNID_PRISIONAL\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LOCAL_FUNC_PRISIONAL_SOCIO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LOCAL_FUNC_TEMPLO_IGREJA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LOCAL_FUNC_CASA_PROFESSOR\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LOCAL_FUNC_GALPAO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_OCUPACAO_GALPAO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LOCAL_FUNC_SALAS_OUTRA_ESC\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LOCAL_FUNC_OUTROS\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_PREDIO_COMPARTILHADO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_AGUA_FILTRADA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_AGUA_REDE_PUBLICA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_AGUA_POCO_ARTESIANO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_AGUA_CACIMBA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_AGUA_FONTE_RIO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_AGUA_INEXISTENTE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ENERGIA_REDE_PUBLICA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ENERGIA_GERADOR\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ENERGIA_OUTROS\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ENERGIA_INEXISTENTE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ESGOTO_REDE_PUBLICA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_',   'spark.sql.sources.schema.part.1'='ESGOTO_FOSSA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ESGOTO_INEXISTENTE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LIXO_COLETA_PERIODICA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LIXO_QUEIMA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LIXO_JOGA_OUTRA_AREA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LIXO_RECICLA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LIXO_ENTERRA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LIXO_OUTROS\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_SALA_DIRETORIA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_SALA_PROFESSOR\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LABORATORIO_INFORMATICA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LABORATORIO_CIENCIAS\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_SALA_ATENDIMENTO_ESPECIAL\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_QUADRA_ESPORTES_COBERTA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_QUADRA_ESPORTES_DESCOBERTA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_QUADRA_ESPORTES\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_COZINHA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_BIBLIOTECA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_SALA_LEITURA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_BIBLIOTECA_SALA_LEITURA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_PARQUE_INFANTIL\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_BERCARIO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_BANHEIRO_FORA_PREDIO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_BANHEIRO_DENTRO_PREDIO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_BANHEIRO_EI\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_BANHEIRO_PNE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_DEPENDENCIAS_PNE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_SECRETARIA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_BANHEIRO_CHUVEIRO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_REFEITORIO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_DESPENSA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ALMOXARIFADO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_AUDITORIO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_PATIO_COBERTO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_PATIO_DESCOBERTO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ALOJAM_ALUNO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ALOJAM_PROFESSOR\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_AREA_VERDE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_LAVANDERIA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_DEPENDENCIAS_OUTRAS\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_SALAS_EXISTENTES\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_SALAS_UTILIZADAS\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_EQUIP_TV\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_EQUIP_VIDEOCASSETE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_EQUIP_DVD\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_EQUIP_PARABOLICA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_EQUIP_COPIADORA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_EQUIP_RETROPROJETOR\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_EQUIP_IMPRESSORA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_EQUIP_IMPRESSORA_MULT\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_EQUIP_SOM\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_EQUIP_MULTIMIDIA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_EQ',   'spark.sql.sources.schema.part.2'='UIP_FAX\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_EQUIP_FOTO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_COMPUTADOR\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_EQUIP_TV\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_EQUIP_VIDEOCASSETE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_EQUIP_DVD\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_EQUIP_PARABOLICA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_EQUIP_COPIADORA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_EQUIP_RETROPROJETOR\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_EQUIP_IMPRESSORA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_EQUIP_IMPRESSORA_MULT\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_EQUIP_SOM\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_EQUIP_MULTIMIDIA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_EQUIP_FAX\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_EQUIP_FOTO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_COMPUTADOR\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_COMP_ADMINISTRATIVO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_COMP_ALUNO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_INTERNET\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_BANDA_LARGA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"QT_FUNCIONARIOS\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ALIMENTACAO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_AEE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_ATIVIDADE_COMPLEMENTAR\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_FUNDAMENTAL_CICLOS\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_LOCALIZACAO_DIFERENCIADA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_MATERIAL_ESP_QUILOMBOLA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_MATERIAL_ESP_INDIGENA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_MATERIAL_ESP_NAO_UTILIZA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_EDUCACAO_INDIGENA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_INDIGENA_LINGUA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_LINGUA_INDIGENA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_BRASIL_ALFABETIZADO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_FINAL_SEMANA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_FORMACAO_ALTERNANCIA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_MEDIACAO_PRESENCIAL\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_MEDIACAO_SEMIPRESENCIAL\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_MEDIACAO_EAD\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ESPECIAL_EXCLUSIVA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_REGULAR\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_EJA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_PROFISSIONALIZANTE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_COMUM_CRECHE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_COMUM_PRE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_COMUM_FUND_AI\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_COMUM_FUND_AF\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_COMUM_MEDIO_MEDIO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_COMUM_MEDIO_INTEGRADO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_COMUM_MEDIO_NORMAL\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ESP_EXCLUSIVA_CRECHE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ESP_EXCLUSIVA_PRE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ESP_EXCLUSIVA_FUND_AI\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ESP_EXCL',   'spark.sql.sources.schema.part.3'='USIVA_FUND_AF\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ESP_EXCLUSIVA_MEDIO_MEDIO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ESP_EXCLUSIVA_MEDIO_INTEGR\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ESP_EXCLUSIVA_MEDIO_NORMAL\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_COMUM_EJA_FUND\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_COMUM_EJA_MEDIO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_COMUM_EJA_PROF\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ESP_EXCLUSIVA_EJA_FUND\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ESP_EXCLUSIVA_EJA_MEDIO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ESP_EXCLUSIVA_EJA_PROF\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_COMUM_PROF\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ESP_EXCLUSIVA_PROF\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='14644043',   'transient_lastDdlTime'='1567454531')
CREATE TABLE escolas_2018_bairro(  cod_mun int,   municipio string,   localizacao string,   esfera_adm string,   cod_inep int,   escola string,   bairro string,   bairro_id string)ROW FORMAT SERDE   'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' WITH SERDEPROPERTIES (   'field.delim'='\;',   'serialization.format'='\;') STORED AS INPUTFORMAT   'org.apache.hadoop.mapred.TextInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/lupaestat.db/escolas_2018_bairro'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='true',   'numFiles'='1',   'skip.header.line.count'='1',   'totalSize'='1213610',   'transient_lastDdlTime'='1567534964')
CREATE TABLE matricula_sudeste(  nu_ano_censo int,   id_aluno string,   id_matricula int,   nu_dia int,   nu_mes int,   nu_ano int,   nu_idade_referencia int,   nu_idade int,   nu_duracao_turma int,   nu_dur_ativ_comp_mesma_rede int,   nu_dur_ativ_comp_outras_redes int,   nu_dur_aee_mesma_rede int,   nu_dur_aee_outras_redes int,   nu_dias_atividade int,   tp_sexo int,   tp_cor_raca int,   tp_nacionalidade int,   co_pais_origem int,   co_uf_nasc int,   co_municipio_nasc int,   co_uf_end int,   co_municipio_end int,   tp_zona_residencial int,   tp_outro_local_aula int,   in_transporte_publico int,   tp_responsavel_transporte int,   in_transp_vans_kombi int,   in_transp_micro_onibus int,   in_transp_onibus int,   in_transp_bicicleta int,   in_transp_tr_animal int,   in_transp_outro_veiculo int,   in_transp_embar_ate5 int,   in_transp_embar_5a15 int,   in_transp_embar_15a35 int,   in_transp_embar_35 int,   in_transp_trem_metro int,   in_necessidade_especial int,   in_cegueira int,   in_baixa_visao int,   in_surdez int,   in_def_auditiva int,   in_surdocegueira int,   in_def_fisica int,   in_def_intelectual int,   in_def_multipla int,   in_autismo int,   in_sindrome_asperger int,   in_sindrome_rett int,   in_transtorno_di int,   in_superdotacao int,   in_recurso_ledor int,   in_recurso_transcricao int,   in_recurso_interprete int,   in_recurso_libras int,   in_recurso_labial int,   in_recurso_braille int,   in_recurso_ampliada_16 int,   in_recurso_ampliada_20 int,   in_recurso_ampliada_24 int,   in_recurso_nenhum int,   tp_ingresso_federais int,   tp_mediacao_didatico_pedago int,   in_especial_exclusiva int,   in_regular int,   in_eja int,   in_profissionalizante int,   tp_etapa_ensino int,   id_turma int,   co_curso_educ_profissional int,   tp_unificada int,   tp_tipo_turma int,   co_entidade int,   co_regiao int,   co_mesorregiao int,   co_microrregiao int,   co_uf int,   co_municipio int,   co_distrito int,   tp_dependencia int,   tp_localizacao int,   tp_categoria_escola_privada int,   in_conveniada_pp int,   tp_convenio_poder_publico int,   in_mant_escola_privada_emp int,   in_mant_escola_privada_ong int,   in_mant_escola_privada_sind int,   in_mant_escola_privada_sist_s int,   in_mant_escola_privada_s_fins int,   tp_regulamentacao int,   tp_localizacao_diferenciada int,   in_educacao_indigena int)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'path'='hdfs://mpmapas-ns/user/hive/warehouse/lupaestat.db/matricula_sudeste') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/lupaestat.db/matricula_sudeste'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='20',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='2',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"NU_ANO_CENSO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"ID_ALUNO\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"ID_MATRICULA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"NU_DIA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"NU_MES\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"NU_ANO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"NU_IDADE_REFERENCIA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"NU_IDADE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"NU_DURACAO_TURMA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"NU_DUR_ATIV_COMP_MESMA_REDE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"NU_DUR_ATIV_COMP_OUTRAS_REDES\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"NU_DUR_AEE_MESMA_REDE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"NU_DUR_AEE_OUTRAS_REDES\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"NU_DIAS_ATIVIDADE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_SEXO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_COR_RACA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_NACIONALIDADE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_PAIS_ORIGEM\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_UF_NASC\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_MUNICIPIO_NASC\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_UF_END\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_MUNICIPIO_END\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_ZONA_RESIDENCIAL\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_OUTRO_LOCAL_AULA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_TRANSPORTE_PUBLICO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_RESPONSAVEL_TRANSPORTE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_TRANSP_VANS_KOMBI\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_TRANSP_MICRO_ONIBUS\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_TRANSP_ONIBUS\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_TRANSP_BICICLETA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_TRANSP_TR_ANIMAL\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_TRANSP_OUTRO_VEICULO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_TRANSP_EMBAR_ATE5\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_TRANSP_EMBAR_5A15\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_TRANSP_EMBAR_15A35\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_TRANSP_EMBAR_35\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_TRANSP_TREM_METRO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_NECESSIDADE_ESPECIAL\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_CEGUEIRA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_BAIXA_VISAO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_SURDEZ\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_DEF_AUDITIVA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_SURDOCEGUEIRA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_DEF_FISICA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_DEF_INTELECTUAL\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_DEF_MULTIPLA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_AUTISMO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_SINDROME_ASPERGER\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_SINDROME_RETT\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_TRANSTORNO_DI\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_SUPERDOTACAO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_RECURSO_LEDOR\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_RECURSO_TRANSCRICAO\",\"type\":\"integer\",\"nullable\":true,\"',   'spark.sql.sources.schema.part.1'='metadata\":{}},{\"name\":\"IN_RECURSO_INTERPRETE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_RECURSO_LIBRAS\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_RECURSO_LABIAL\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_RECURSO_BRAILLE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_RECURSO_AMPLIADA_16\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_RECURSO_AMPLIADA_20\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_RECURSO_AMPLIADA_24\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_RECURSO_NENHUM\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_INGRESSO_FEDERAIS\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_MEDIACAO_DIDATICO_PEDAGO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_ESPECIAL_EXCLUSIVA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_REGULAR\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_EJA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_PROFISSIONALIZANTE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_ETAPA_ENSINO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"ID_TURMA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_CURSO_EDUC_PROFISSIONAL\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_UNIFICADA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_TIPO_TURMA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_ENTIDADE\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_REGIAO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_MESORREGIAO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_MICRORREGIAO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_UF\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_MUNICIPIO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"CO_DISTRITO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_DEPENDENCIA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_LOCALIZACAO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_CATEGORIA_ESCOLA_PRIVADA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_CONVENIADA_PP\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_CONVENIO_PODER_PUBLICO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_MANT_ESCOLA_PRIVADA_EMP\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_MANT_ESCOLA_PRIVADA_ONG\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_MANT_ESCOLA_PRIVADA_SIND\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_MANT_ESCOLA_PRIVADA_SIST_S\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_MANT_ESCOLA_PRIVADA_S_FINS\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_REGULAMENTACAO\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"TP_LOCALIZACAO_DIFERENCIADA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"IN_EDUCACAO_INDIGENA\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='1193676561',   'transient_lastDdlTime'='1567454147')
CREATE TABLE projecoes(  cod_mun int,   mun string,   ano int,   pop_0_3 int,   pop_4_5 int,   pop_6_14 int,   pop_16 int,   pop_15_17 int)ROW FORMAT SERDE   'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' WITH SERDEPROPERTIES (   'charset'='UTF-8',   'path'='hdfs://mpmapas-ns/user/hive/warehouse/lupaestat.db/projecoes') STORED AS INPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT   'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'LOCATION  'hdfs://mpmapas-ns/user/hive/warehouse/lupaestat.db/projecoes'TBLPROPERTIES (  'COLUMN_STATS_ACCURATE'='false',   'numFiles'='1',   'numRows'='-1',   'rawDataSize'='-1',   'spark.sql.create.version'='2.3.0.cloudera2',   'spark.sql.sources.provider'='parquet',   'spark.sql.sources.schema.numParts'='1',   'spark.sql.sources.schema.part.0'='{\"type\":\"struct\",\"fields\":[{\"name\":\"cod_mun\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"mun\",\"type\":\"string\",\"nullable\":true,\"metadata\":{}},{\"name\":\"ano\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"pop_0_3\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"pop_4_5\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"pop_6_14\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"pop_16\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}},{\"name\":\"pop_15_17\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}}]}',   'totalSize'='5047',   'transient_lastDdlTime'='1567452045')
