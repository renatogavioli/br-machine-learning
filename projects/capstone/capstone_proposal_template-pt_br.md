# Nanodegree Engenheiro de Machine Learning
## Proposta de projeto final
Renato Rosafa Gavioli
03 de julho de 2018

## Reconhecimento de Atividade Humana Usando Dados de Smartphone

### Histórico do assunto

Com o advento dos smartphones e seu custo cada vez menor, é cada vez mais comum as pessoas possuírem dispositivos em seus bolsos dotados de diversos sensores inerciais como acelerômetros e giroscópios. Com isso, tornam-se disponíveis dados sobre os padrões de deslocamento dos usuários que permitem uma série de insights. Associados a tecnologias como GPS e smartwatches...

A evolução da pirâmide etária brasileira vem apontando para um gradual envelhecimento da população. Este envelhecimento, caracterizado por um aumento na quntidade de idosos na população, é decorrente do crescente grau de urbanização do desenvolvimento econômico e humano do país [https://www.ibge.gov.br/apps/populacao/projecao/, acesso em 09.07.2018]. 

Idosos são frequentemente vítimas de quedas. Além dos problemas médicos, as quedas apresentam custo social, econômico e psicológico enormes, aumentando a dependência e a institucionalização. Estima-se que há uma queda para um em cada três indivíduos com mais de 65 anos e, que um em vinte daqueles que sofreram uma queda sofram uma fratura ou necessitem de internação. Dentre os mais idosos, com 80 anos e mais, 40% caem a cada ano. Dos que moram em asilos e casas de repouso, a freqüência de quedas é de 50%. A prevenção de quedas é tarefa difícil devido a variedade de fatores que as predispõem.[http://bvsms.saude.gov.br/bvs/dicas/184queda_idosos.html, acesso em 09.07.2018] 

Com o advento e a democratização de _smartphones_ dotados de inúmeros sensores [https://www.uni-weimar.de/kunst-und-gestaltung/wiki/images/Zeitmaschinen-smartphonesensors.pdf, acesso em 09.07.2018], surgiram diversas aplicações de sensoreament que fazem uso destes sensores e das capacidades de processamento e transmissão dos _smartphones_, para aquisição e processamento de dados, exibição e comunicação. 

O monitoramento e análise dos dados proveniente de sensores inerciais como acelerômetros e giroscópios podem permitir prever qual o padrão de atividade e movimentação do usuário, utilizando um algoritmo de classificação supervisionada. Com isso, torna-se possível a criação de aplicações para identificar uma queda em um idoso, permitindo ações de socorro mais rápidas, além de um cenário mais claro da distribuição de quedas, suas causas e contextos como subsídio para ações preventivas. Este tipo de estratégia já vem sendo aplicada para identificação de crises epilépticas [https://www.researchgate.net/profile/Khaled_Elleithy/publication/322921138_Smart_Phone_Application_Development_for_Monitoring_Epilepsy_Seizure_Detection_based_on_EEG_signal_Classification/links/5a76327145851541ce588920/Smart-Phone-Application-Development-for-Monitoring-Epilepsy-Seizure-Detection-based-on-EEG-signal-Classification.pdf?origin=publication_list, acesso em 09.07.2018]. 

### Descrição do problema

O problema a ser resolvido é um problema de classificação supervisionada. Dado uma observação ou um conjunto de observações provenientes de acelerômetros e sensores de um smartphone, necessitamos saber qual a atividade praticada pelo usuário daquele smartphone com algum grau de certeza. Podemos medir o quão bem resolvemos este problema a partir das taxas de acertos - numa situação de classificação binária (queda vs. não-queda), o número de falsos negativos não deveria ser um problema, e a métrica principal de revocação (recall), e não de precisão, torna-se mais apropriada. O problema é replicável, uma vez que o conjunto de dados encontra-se publicamente disponível.

### Conjuntos de dados e entradas

Serão utilizados os dados _Human Activity Recognition Using Smartphones Data Set_, disponíveis em [https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones]. Este conjunto foi produzido por um experimento no qual diversos voluntários praticaram diversas atividades com um aparelho preso à cintura - as informações geradas pelos acelerômetros foram preprocessadas e disponibilizadas. O conjunto de dados compreende cerca de 10000 observações de 561 caraterísticas (features), relacionadas a 7 atividades diferentes (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING). Será tomada como atividade alvo a de deitar (LAYING), que pode corresponder a uma queda, e as demais serão agregadas, representado uma situação na qual não haja queda.

As características compreendem dados de acelerômetro e giroscópio em 3 direções. Os dados foram preprocessados de modo que, dentre as características, estao máximos, mínimos, médias e dados convertidos em termos de aceleração, aceleração gravitacional, energia - acredito que uma boa parte destes features podem estar de certo modo relacionados, podendo ser reduzidos.

Uma limitação do trabalho é o fato de o conjunto de dados ter sido gerado por voluntários entre 19 e 48 anos, cujo padrão motor provavelmente é bastante diferente do de idosos. Entende-se que o conjunto pode ser suficiente para o desenvolvimento do método e das estratégias de tratamento de dados e aprendizagem, mas que um novo conjunto de dados deveria ser gerado com uma amostra da população de interesse, e este conjunto utilizado para uma aplicação em produção.

### Descrição da solução
_(aprox. 1 parágrafo)_

Nesta seção, descreva claramente uma solução para o problema. A solução deve ser relevante ao assunto do projeto e adequada ao(s) conjunto(s) ou entrada(s) proposto(s). Descreva a solução detalhadamente, de forma que fique claro que o problema é quantificável (a solução pode ser expressa em termos matemáticos ou lógicos), mensurável (a solução pode ser medida por uma métrica e claramente observada) e replicável (a solução pode ser reproduzida e ocorre mais de uma vez).

### Modelo de referência (benchmark)
_(aproximadamente 1-2 parágrafos)_

Nesta seção, forneça os detalhes de um modelo ou resultado de referência que esteja relacionado ao assunto, definição do problema e solução proposta. Idealmente, o resultado ou modelo de referência contextualiza os métodos existentes ou informações conhecidas sobre o assunto e problema propostos, que podem então ser objetivamente comparados à solução. Descreva detalhadamente como o resultado ou modelo de referência é mensurável (pode ser medido por alguma métrica e claramente observado).

### Métricas de avaliação
_(aprox. 1-2 parágrafos)_

Nesta seção, proponha ao menos uma métrica de avaliação que pode ser usada para quantificar o desempenho tanto do modelo de benchmark como do modelo de solução apresentados. A(s) métrica(s) de avaliação proposta(s) deve(m) ser adequada(s), considerando o contexto dos dados, da definição do problema e da solução pretendida. Descreva como a(s) métrica(s) de avaliação pode(m) ser obtida(s) e forneça um exemplo de representação matemática para ela(s) (se aplicável). Métricas de avaliação complexas devem ser claramente definidas e quantificáveis (podem ser expressas em termos matemáticos ou lógicos)

### Design do projeto
_(aprox. 1 página)_

Nesta seção final, sintetize um fluxo de trabalho teórico para obtenção de uma solução para o problema em questão. Discuta detalhadamente quais estratégias você considera utilizar, quais análises de dados podem ser necessárias de antemão e quais algoritmos serão considerados na sua implementação. O fluxo de trabalho e discussão propostos devem estar alinhados com as seções anteriores. Adicionalmente, você poderá incluir pequenas visualizações, pseudocódigo ou diagramas para auxiliar na descrição do design do projeto, mas não é obrigatório. A discussão deve seguir claramente o fluxo de trabalho proposto para o projeto de conclusão.

-----------

**Antes de enviar sua proposta, pergunte-se. . .**

- A proposta que você escreveu segue uma estrutura bem organizada, similar ao modelo de projeto?
- Todas as seções (em especial, **Descrição da solução** e **Design do projeto**) estão escritas de uma forma clara, concisa e específica? Existe algum termo ou frase ambígua que precise de esclarecimento?
- O público-alvo de seu projeto será capaz de entender sua proposta?
- Você revisou sua proposta de projeto adequadamente, de forma a minimizar a quantidade de erros gramaticais e ortográficos?
- Todos os recursos usados neste projeto foram corretamente citados e referenciados?
