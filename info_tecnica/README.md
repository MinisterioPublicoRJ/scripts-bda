# info_tecnica
Repositório de script para extrair e mover os arquivos PDFs de Informação Técnica GATE para o HDFS

## Configuração e ambiente

O projeto utiliza decouple para variáveis de configuração e ambiente, é necessário exportar no ambiente as seguintes variáveis:

```
ORACLE_SERVER=(Servidor do banco oracle)
DIR_FILES_PDF=(Diretorio de destino no HDFS dos PDFs)
HDFS_SERVER=(Servidor do HDFS)
HDFS_USER=(Usuario no HDFS)
ORACLE_USER=(Usuario do banco oracle)
ORACLE_PASSWORD=(Senha do banco oracle)
```

## Para executar o projeto:

###### make install -> Para baixar as dependencias e configura-las numa pasta local

###### make -> Para executar o script shell que incia o processo de extração e transferencias dos PDFs para o repositorio do GATE no HDFS
