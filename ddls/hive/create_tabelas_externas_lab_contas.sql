SET hive.mapred.supports.subdirectories=true;
SET mapreduce.input.fileinputformat.input.dir.recursive=true;

drop table staging.lc_cnpj;
drop table staging.lc_cpf;
drop table staging.lc_socio;
drop table staging.lc_ppe;
drop table staging.lc_vinculo_trabalhista;
drop table staging.detran_regcivil;
drop table staging.detran_multa;
drop table staging.detran_veiculo;


create external table staging.lc_cnpj (
	 NUM_CNPJ   varchar (14)  ,
	 IND_MATRIZ_FILIAL   decimal (1, 0)  ,
	 NOME   varchar (150)  ,
	 NOME_FANTASIA   varchar (55)  ,
	 COD_TIPO_LOGRADOURO   varchar (6)  ,
	 TIPO_LOGRADOURO   varchar (20)  ,
	 DESCR_LOGRADOURO   varchar (60)  ,
	 NUM_LOGRADOURO   varchar (6)  ,
	 DESCR_COMPLEMENTO_LOGRADOURO   varchar (156)  ,
	 NOME_BAIRRO   varchar (50)  ,
	 NUM_CEP   decimal (8, 0)  ,
	 COD_MUNICIPIO   varchar (4)  ,
	 NOME_MUNICIPIO   varchar (50)  ,
	 SIGLA_UF   varchar (2)  ,
	 NUM_DDD1   decimal (4, 0)  ,
	 NUM_TELEFONE1   decimal (8, 0)  ,
	 NUM_DDD2   decimal (4, 0)  ,
	 NUM_TELEFONE2   decimal (8, 0)  ,
	 NUM_DDD_FAX   decimal (4, 0)  ,
	 NUM_FAX   decimal (8, 0)  ,
	 NOME_CIDADE_EXTERIOR   varchar (55)  ,
	 NOME_PAIS   varchar (70)  ,
	 DESCR_EMAIL   varchar (115)  ,
	 SE_TEM_SOCIO   varchar (1)  ,
	 COD_SITUACAO_CADASTRAL_CNPJ   decimal (2, 0)  ,
	 DATA_SITUACAO_CADASTRAL   timestamp   ,
	 COD_MOTIVO_SITUACAO_CADASTRAL   decimal (2, 0)  ,
	 COD_NATUREZA_JURIDICA   decimal (4, 0)  ,
	 COD_ATIVIDADE_ECON_PRINCIPAL   decimal (7, 0)  ,
	 DATA_ABERTURA_ESTABELECIMENTO   timestamp   ,
	 DATA_INCLUSAO_SIMPLES   timestamp   ,
	 DATA_EXCLUSAO_SIMPLES   timestamp   ,
	 COD_PORTE_EMPRESA   decimal (2, 0)  ,
	 COD_QUALIFICACAO_TRIBUTARIA   decimal (2, 0)  ,
	 NUM_NIRE   varchar (11)  ,
	 NUM_CPF_RESPONSAVEL   varchar (11)  ,
	 NOME_RESPONSAVEL   varchar (60)  ,
	 COD_QUALIFICACAO_RESPONSAVEL   decimal (2, 0)  ,
	 IND_TIPO_OPERACAO_SUCEDIDA   decimal (1, 0)  ,
	 NUM_CNPJ_SUCEDIDA   varchar (14)  ,
	 DATA_EVENTO_SUCEDIDA   timestamp   ,
	 IND_TIPO_OPERACAO_SUCESSORA   decimal (1, 0)  ,
	 DATA_EVENTO_SUCESSORA   timestamp   ,
	 NUM_CNPJ_SUCESSORA   varchar (14)  ,
	 VALOR_CAPITAL_SOCIAL   decimal (17, 0)  ,
	 COD_ORGAO_ADUANEIRO   decimal (7, 0)  
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
stored as textfile
location '/user/mpmapas/staging/labcontas/lc_cnpj';



create external table staging.lc_cpf (
     NUM_CPF   char (11)  ,
	 NOME   varchar (100)  ,
	 DATA_NASCIMENTO   timestamp   ,
	 IND_SEXO   char (1)  ,
	 NOME_MAE   varchar (60)  ,
	 NUM_TITULO_ELEITOR   varchar (12)  ,
	 TIPO_LOGRADOURO   varchar (15)  ,
	 DESCR_LOGRADOURO   varchar (40)  ,
	 NUM_LOGRADOURO   varchar (6)  ,
	 DESCR_COMPLEMENTO_LOGRADOURO   varchar (21)  ,
	 NOME_BAIRRO   varchar (19)  ,
	 NUM_CEP   varchar (8)  ,
	 NOME_MUNICIPIO   varchar (50)  ,
	 SIGLA_UF   varchar (2)  ,
	 NUM_DDD   varchar (4)  ,
	 NUM_TELEFONE   varchar (8)  ,
	 NUM_FAX   varchar (8)  ,
	 SE_ESTRANGEIRO   varchar (1)  ,
	 NOME_PAIS_NACIONALIDADE   varchar (60)  ,
	 COD_SITUACAO_CADASTRAL   varchar (1)  ,
	 DESCR_SITUACAO_CADASTRAL   varchar (40)  ,
	 DATA_SITUACAO_CADASTRAL   timestamp   ,
	 DATA_INSCRICAO   timestamp   ,
	 ANO_OBITO   varchar (4)  ,
	 ANO_ULTIMA_ENTREGA_DECLARACAO   varchar (4)  
)
ROW FORMAT SERDE 
	  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
WITH SERDEPROPERTIES ( 
	  'field.delim'=',', 
	  'serialization.format'=',',
	  'quoteChar'='"',
	  'timestamp.formats'='yyyy-MM-dd'
)
stored as textfile
location '/user/mpmapas/staging/labcontas/lc_cpf';


create external table staging.lc_vinculo_trabalhista (
	cnpj varchar(14) ,
	cpf varchar(11) ,
	dt_inicio timestamp ,
	dt_fim timestamp ,
	tp_vinculo varchar(255) ,
	vinc_ativo int 
)
ROW FORMAT SERDE 
	  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
WITH SERDEPROPERTIES ( 
	  'field.delim'=',', 
	  'serialization.format'=',',
	  'timestamp.formats'='yyyy-MM-dd'
)
stored as textfile
location '/user/mpmapas/staging/labcontas/lc_vinculo_trabalhista';



create external table staging.lc_socio (
	cnpj varchar(14) ,
	tipo varchar(2) ,
	cpf_socio varchar(11) ,
	cnpj_socio varchar(14) ,
	dt_inicio timestamp ,
	dt_fim timestamp ,
	percentual decimal(5, 2) ,
	tipo_socio varchar(60) ,
	vinc_ativo int 
)
ROW FORMAT SERDE 
	  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
WITH SERDEPROPERTIES ( 
	  'field.delim'=',', 
	  'serialization.format'=',',
	  'timestamp.formats'='yyyy-MM-dd'
)
stored as textfile
location '/user/mpmapas/staging/labcontas/lc_socio';



drop table staging.lc_embarcacao;

create external table staging.lc_embarcacao(
    ID_EMBARCACAO int ,
	DS_NOME_EMBARCACAO varchar(100) ,
	TIPO_EMBARCACAO varchar(60) ,
	ANO_CONSTRUCAO smallint ,
	NR_COMPRIMENTO decimal(9, 4) ,
	NR_INSCRICAO varchar(14) ,
	SITUACAO_EMBARCACAO varchar(30) ,
	DT_INSCRICAO_EMB timestamp ,
	ORGAO_INSCRICAO varchar(13) ,
	DS_CIDADE_ORGAO varchar(25) ,
	CPF_CNPJ varchar(14) ,
	PROPR_ARMADOR_AFRET_ATUAL varchar(3) ,
	ULT_OBS_IMP_DOC_EMB_GR_PORTE varchar(200) ,
	DATA_AQUISICAO timestamp ,
	ULT_LOCAL_AQUISICAO_PROP_ATUAL varchar(50) ,
	ULT_VALOR_AQUISICAO_PROP_ATUAL decimal(13, 2) ,
	CONSTRUTOR_CASCO varchar(30) 
)
ROW FORMAT SERDE 
	  'org.apache.hadoop.hive.serde2.OpenCSVSerde' 
WITH SERDEPROPERTIES ( 
	  'separatorChar'=',',
	  'quoteChar'='"'
)
stored as textfile
location '/user/mpmapas/staging/labcontas/lc_embarcacao';

drop table staging.detran_regcivil;

create external table staging.detran_regcivil(
    BASE varchar(50),
    NU_RG varchar(50),
    DT_EXPEDICAO_CARTEIRA timestamp,
    NO_CIDADAO varchar(255),
    NO_PAICIDADAO varchar(255),
    NO_MAECIDADAO varchar(255),
    NATURALIDADE varchar(100),
    DT_NASCIMENTO timestamp,
    DOCUMENTO_ORIGEM varchar(255),
    NU_CPF varchar(11),
    ENDERECO varchar(255),
    BAIRRO varchar(255),
    MUNICIPIO varchar(255),
    UF varchar(2),
    CEP varchar(8)
)
ROW FORMAT SERDE 
	  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
WITH SERDEPROPERTIES ( 
	  'field.delim'='|',
	  'timestamp.formats'='dd/MM/yyyy',
      'serialization.encoding'='LATIN1'
)
stored as textfile
location '/user/mpmapas/staging/detran/detran_regcivil'
TBLPROPERTIES ("skip.header.line.count"="1");


create external table staging.detran_multa(
    aa_inf int,
    cd_org_aut string,
    ds_org_aut string,
    autoinfra string,
    id_ntf_ar_aut_sm string,
    pl_vei_inf string,
    cd_inf string,
    dv_cd_inf string,
    desd_cd_inf string,
    ds_inf_tab string,
    tp_enq_tab string,
    pto_inf_tab  string,
    cd_cls_agt string,
    nu_agt_inf string,
    dt_inf timestamp,
    hr_inf int,
    localinfra string,
    tve_tab_descricao_marca string,
    descricao_especie string,
    descricao_categorias string,
    descricao_tipo string,
    descricao_cor string,
    dt_status_aut_sm timestamp,
    cd_status_aut_sm string,
    vl_mul_real_inf string,
    tp2 string,
    ident2 string,
    nm_con_inf string,
    nm_log_end_inf string,
    nu_end_inf string,
    cp_end_inf string,
    nu_cep_end_inf string,
    cd_mun_end_inf string,
    descricao_municipio2 string,
    ve_per_inf int,
    ve_con_inf int,
    ve_afe_inf int,
    lo_med_inf string
)
ROW FORMAT SERDE 
	  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
WITH SERDEPROPERTIES ( 
	  'field.delim'=';',
	  'timestamp.formats'='yyyyMMdd',
      'serialization.encoding'='LATIN1'
)
stored as textfile
location '/user/mpmapas/staging/detran/detran_multa'
TBLPROPERTIES ("skip.header.line.count"="1");


create external table staging.detran_veiculo(
    PLACA string,
    PLACA_ANTERIOR string,
    RENAVAM string,
    CHASSI string,
    DESCRICAO_ESPECIE string,
    DESCRICAO_TIPO string,
    MARCA_MODELO string,
    DESCRICAO_COMBUSTIVEL string,
    DESCRICAO_CATEGORIAS string,
    DESCRICAO_COR string,
    DESCRICAO_MUNICIPIO_EMP string,
    FABRIC int,
    MODELO int,
    MOTOR string,
    TVE_CAPACIDADE_PASSAG string,
    H_P string,
    CILINDRADA string,
    TVE_FAIXA_IPVA string,
    ULTIMO_LIC int,
    PROPRIETARIO string,
    CPFCGC string,
    TIPO_CIC int,
    ENDERECO string,
    NUMERO string,
    COMPLEMENTO string,
    CEP string,
    MUNICIPIO string,
    MUNIC_ENDERECO string,
    PROPRIETARIO_GRAVAME string,
    CPFCGC_GRAVAME string,
    TIPO_CIC_GRAVAME string,
    ENDERECO_GRAVAME string,
    NUMERO_GRAVAME string,
    COMPLEMENTO_GRAVAME string,
    CEP_GRAVAME string,
    MUNICIPIO_GRAVAME string,
    MUNIC_ENDERECO_GRAVAME string,
    NOME_CV string,
    TIPO_CIC_CV string,
    CPFCGC_CV string,
    ENDERECO_CV string,
    NUMERO_CV string,
    COMPLEMENTO_CV string,
    CEP_CV string,
    MUNICIPIO_CV string,
    MUNIC_ENDERECO_CV string,
    BAIRRO_CV string,
    UF_CV string,
    DATA_VENDA_CV string
)
ROW FORMAT SERDE 
	  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
WITH SERDEPROPERTIES ( 
	  'field.delim'=';',
	  'timestamp.formats'='yyyyMMdd',
      'serialization.encoding'='LATIN1'
)
stored as textfile
location '/user/mpmapas/staging/detran/detran_veiculo'
TBLPROPERTIES ("skip.header.line.count"="1");


create external table staging.lc_ppe (
	cpf varchar(11) ,
	nome varchar(255),
	cargo varchar(255)
)
ROW FORMAT SERDE 
	  'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe' 
WITH SERDEPROPERTIES ( 
	  'field.delim'=',', 
	  'serialization.format'=',',
	  'timestamp.formats'='yyyy-MM-dd'
)
stored as textfile
location '/user/mpmapas/staging/labcontas/lc_ppe/';