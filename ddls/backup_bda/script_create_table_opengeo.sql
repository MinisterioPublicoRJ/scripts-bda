CREATE TABLE financeiro(  id int,   codigo_do_imovel string,   natureza_do_imovel string,   tipo_de_imovel string,   craai string,   municipio string,   predio string,   complemento_imobiliario string,   predio_complemento string,   area_do_layout decimal(12,2),   codigo_cc string,   centro_de_custos string,   tipo_de_orgao string,   endereco_principal string,   fator_de_custo_orgao decimal(12,2),   fator_de_custo_complemento decimal(12,2),   fator_de_custo_imovel decimal(12,2),   fator_de_custo_predio decimal(12,2),   fator_de_custo_municipio decimal(12,2),   fator_de_custo_craai decimal(12,2),   item_de_custo string,   tipo_de_custo string,   competencia timestamp,   soma_cc decimal(12,2),   soma_complemento decimal(12,2),   soma_imovel decimal(12,2),   soma_predio decimal(12,2),   soma_municipio decimal(12,2),   soma_craai decimal(12,2),   total decimal(12,2)) STORED AS PARQUET;
CREATE TABLE lupa_cod_municipio(  cod_municipio int,   municipio string) STORED AS PARQUET;
CREATE TABLE lupa_dados_gerais_municipio(  cod_municipio int,   municipio string,   criacao string,   aniversario string) STORED AS PARQUET;
CREATE TABLE lupa_governantes_rj(  nome string,   cpf string,   localidade string,   cod_ibge int,   ano_eleicao int,   base64_foto string,   url_tse string) STORED AS PARQUET;
CREATE TABLE lupa_orgaos_mprj(  id bigint,   codigo_mgo int,   orgao string,   municipio string,   codigo_municipio int,   codigo_craai int,   craai string,   endereco string,   complemento string,   natureza string,   tipo string,   geom string,   longitude double,   latitude double) STORED AS PARQUET;
CREATE TABLE lupa_prefeituras(  cod_craai string,   nome_craai string,   municipio string,   cod_municipio int,   gentilico string,   prefeito string,   vice_prefeito string,   site string,   telefone string,   endereco string,   fonte string) STORED AS PARQUET;
CREATE TABLE municipio_flag(  cod_municipio int,   municipio string,   criacao string,   aniversario string,   flag string) STORED AS PARQUET;
