CREATE TABLE base_caracteristicas_vitimas(  tipo_cabelo int,   cor_cabelo int,   cor_olho int,   altura int,   cor_pele int,   biotipo int,   tipo_sangue int,   caract_indiv int,   parte_corpo int,   identificador_sinalid string) STORED AS PARQUET;
CREATE TABLE pertinencia3_cor_pele(  v1 double,   v2 double,   v3 double,   vitima string) STORED AS PARQUET;
CREATE TABLE pertinencia4(  v1 double,   v2 double,   v3 double,   v4 double,   vitima string) STORED AS PARQUET;
CREATE TABLE pertinencia5_altura(  v1 double,   v2 double,   v3 double,   v4 double,   v5 double,   vitima string) STORED AS PARQUET;
CREATE TABLE vizinho3_cor_pele(  cluster double,   neighbor double,   sil_width double,   vitima string) STORED AS PARQUET;
CREATE TABLE vizinho4(  cluster double,   neighbor double,   sil_width double,   vitima string) STORED AS PARQUET;
CREATE TABLE vizinho5_altura(  cluster double,   neighbor double,   sil_width double,   vitima string) STORED AS PARQUET;
