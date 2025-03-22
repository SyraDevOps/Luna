# Luna

![Luna](img/Luna.png)
 
Luna 1.00 Model by Syra


SyRa_LUNA
Este projeto implementa um sistema de aprendizado de máquina para criar, treinar e utilizar um modelo de IA para responder perguntas com base em um banco de dados de conhecimento. O sistema utiliza PyTorch para a criação e treinamento do modelo, e SQLite para armazenar os dados de treinamento.

Estrutura do Projeto
ext.py: Arquivo principal para iniciar o sistema.

NeuronChat.py: Arquivo principal para iniciar o chat com o modelo de IA.

Neutron: Diretório contendo os módulos principais do sistema.

brain.py: Implementa a classe Cerebro que gerencia o modelo de IA e o banco de dados.

functions.py: Funções auxiliares para carregar dados e treinar modelos.

functionspreditc.py: Funções para carregar modelos e fazer previsões.

Mnu.py: Interface de linha de comando para interagir com o sistema.

NeuronChat.py: Implementa a classe Cerebro para o chat.

Datasets: Diretório contendo arquivos CSV de exemplo para treinamento.

Requisitos

Instale as dependências listadas no arquivo requirements.txt:

pip install -r requirements.txt

Passos para Criar, Treinar e Utilizar o Modelo

# 1. Criar o Modelo

Para criar um novo modelo, execute o arquivo ext.py:

python ext.py

Você verá a seguinte interface:

=== Sistema de Aprendizado com Contexto ===
Comandos especiais:
/contexto [nome] - Define um contexto
/csv [arquivo] - Treina com CSV
/compromisso [desc data hora] - Agenda compromisso
/trocar [modelo] - Troca de modelo
/treinar [épocas] - Treina com todos os neurônios
/sair - Encerra o programa

Modelos disponíveis:
 - default
Nome do modelo:

Digite o nome do modelo que deseja criar ou pressione Enter para usar o modelo padrão.

# 2. Treinar o Modelo

Para treinar o modelo com um arquivo CSV, use o comando /csv [arquivo]:

/csv Datasets/DataExample01.csv

Você será solicitado a inserir o número de épocas para o treinamento. Digite o número desejado e pressione Enter.

# 3. Utilizar o Chat

Para iniciar o chat com o modelo de IA, execute o arquivo NeuronChat.py:

python NeuronChat.py

Você verá a seguinte interface:

=== Chat com Modelo de IA ===
Modelos disponíveis:
 - default
Nome do modelo:

Digite o nome do modelo que deseja usar para o chat ou pressione Enter para usar o modelo padrão.

# 4. Interagir com o Chat
Digite suas perguntas e o sistema responderá com base no modelo treinado. Para sair do chat, digite /sair.


Comandos Especiais

/contexto [nome]: Define um contexto para as perguntas.
/csv [arquivo]: Treina o modelo com um arquivo CSV.
/compromisso [desc data hora]: Agenda um compromisso.
/trocar [modelo]: Troca para outro modelo.
/treinar [épocas]: Treina o modelo com todos os neurônios.
/sair: Encerra o programa.

# Estrutura dos Arquivos CSV

Os arquivos CSV devem conter as seguintes colunas:

pergunta: A pergunta a ser feita ao modelo.
resposta: A resposta correspondente à pergunta.
contexto: O contexto da pergunta (pode ser vazio).

Exemplo de arquivo CSV:

pergunta,resposta,contexto
"Qual seu nome?","Meu nome é Luna.","Identidade"
"Como você se chama?","Meu nome é Luna.","Identidade"

# Contribuição
Sinta-se à vontade para contribuir com melhorias para este projeto. Envie pull requests ou abra issues para discutir mudanças.
