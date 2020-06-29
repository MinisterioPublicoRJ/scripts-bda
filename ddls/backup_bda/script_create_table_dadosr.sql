CREATE TABLE 17102018_teste_analise_corregedoria(  data string,   pj string,   clusters string,   noticia_fato string,   p_adm string,   inqueritos_civis string,   baixas_atraso string) STORED AS PARQUET;
CREATE TABLE cluster_39(  promotoria string,   cluster int) STORED AS PARQUET;
CREATE TABLE dados(  alerta string,   orgao string,   qtd int,   cdorgao double,   tipo string) STORED AS PARQUET;
CREATE TABLE dados_cluster_setembro(  classe string,   atribuicao string,   orgao_responsvel string,   responsavel string,   dt_criacao_orgao string) STORED AS PARQUET;
CREATE TABLE dadosalerta(  alerta string,   orgao string,   qtd int,   cdorgao double) STORED AS PARQUET;
CREATE TABLE historico_acervo_promotoria(  data string,   pj string,   entradas string,   saidas string,   acervo string,   saldo string) STORED AS PARQUET;
CREATE TABLE previsao_entradas_teste(  pj string,   point_forecast double,   lo_80 double,   hi_80 double,   lo_95 double,   hi_95 double) STORED AS PARQUET;
CREATE TABLE previsao_saida_teste(  pj string,   point_forecast double,   lo_80 double,   hi_80 double,   lo_95 double,   hi_95 double) STORED AS PARQUET;
CREATE TABLE result_cluster(  pj string,   nome_orgao string,   atribuicao string,   qtd double,   cluster int) STORED AS PARQUET;
CREATE TABLE result_pacote_atribuicao(  cdorgao double,   rcor_ds_relatorio string,   data string,   grupo string,   qt_feitos double) STORED AS PARQUET;
CREATE TABLE tabela_final_cluster123(  dt double,   cdorgao string,   entradas double,   saidas double,   saldo double,   tipo string,   orgao string,   cluster int) STORED AS PARQUET;
CREATE TABLE tabela_finall_cluster(  dt string,   cdorgao string,   entradas string,   saidas string,   saldo string,   tipo string,   orgao string,   cluster int) STORED AS PARQUET;
CREATE TABLE tabelaa_final(  dt string,   cdorgao string,   entradas string,   saidas string,   saldo string,   tipo string,   orgao string) STORED AS PARQUET;