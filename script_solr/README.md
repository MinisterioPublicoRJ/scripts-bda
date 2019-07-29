#  Scripts para criar as entidades no SOLR
Repositório de script para criar e carregar os dados das tabelas HIVE para as entidades do SOLR.

## Configuração e ambiente

git clone url_projeto

Edite o arquivo data-config.xml:

Na tag entity crie um nome para identificar sua entidade: Ex: pessoa_juridica

Na tag query crie o seu sql com os dados que deseja incluir na entidade do SOLR


         <document>
                  <entity name="pessoa_juridica"
                     query="SELECT * from table">
                 </entity>
         </document>

Edite o arquivo schema.xml:

Na tag field crie seu campo de acordo com o tipo de dado da sua tabela:

**Obs: Voce pode criar quantos campos necessários, porém os campos _version_ e _root_ são obrigatorios**

         <fields>
                 <field name="_version_" type="long" indexed="true" stored="true" multiValued="false" />
                 <field name="_root_" type="string" indexed="true" stored="false" docValues="false" multiValued="false"/>
                 <field name="exemplo_nome_pessoa" type="string" indexed="true" stored="true" required="true"          multiValued="false" />
          </fields>

## Para executar o projeto:

###### ./create_solr_process.sh  `<server> <base-path> <type-process> <name-entity>`
         
base_path = /home/teste > onde se encontra a pasta conf 
         
Ex: ./create_solr_process.sh datanode07.mp.gov.br base_path 1 pessoa_juridica
