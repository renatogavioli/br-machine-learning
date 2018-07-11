# Nanodegree Engenheiro de Machine Learning
## Proposta de projeto final
Renato Rosafa Gavioli
10 de julho de 2018

## Reconhecimento de Quedas Usando Dados de Smartphone


### Histórico do assunto

Com o advento dos smartphones e seu custo cada vez menor, é cada vez mais comum as pessoas possuírem dispositivos em seus bolsos dotados de diversos sensores inerciais como acelerômetros e giroscópios. Com isso, tornam-se disponíveis dados sobre os padrões de deslocamento dos usuários que permitem uma série de insights. Associados a tecnologias como GPS e smartwatches...

A evolução da pirâmide etária brasileira vem apontando para um gradual envelhecimento da população. Este envelhecimento, caracterizado por um aumento na quntidade de idosos na população, é decorrente do crescente grau de urbanização do desenvolvimento econômico e humano do país [https://www.ibge.gov.br/apps/populacao/projecao/, acesso em 09.07.2018].

Idosos são frequentemente vítimas de quedas. Além dos problemas médicos, as quedas apresentam custo social, econômico e psicológico enormes, aumentando a dependência e a institucionalização. Estima-se que há uma queda para um em cada três indivíduos com mais de 65 anos e, que um em vinte daqueles que sofreram uma queda sofram uma fratura ou necessitem de internação. Dentre os mais idosos, com 80 anos e mais, 40% caem a cada ano. Dos que moram em asilos e casas de repouso, a freqüência de quedas é de 50%. A prevenção de quedas é tarefa difícil devido a variedade de fatores que as predispõem.[http://bvsms.saude.gov.br/bvs/dicas/184queda_idosos.html, acesso em 09.07.2018]

Com o advento e a democratização de _smartphones_ dotados de inúmeros sensores [https://www.uni-weimar.de/kunst-und-gestaltung/wiki/images/Zeitmaschinen-smartphonesensors.pdf, acesso em 09.07.2018], surgiram diversas aplicações de sensoreament que fazem uso destes sensores e das capacidades de processamento e transmissão dos _smartphones_, para aquisição e processamento de dados, exibição e comunicação.

O monitoramento e análise dos dados proveniente de sensores inerciais como acelerômetros e giroscópios podem permitir prever qual o padrão de atividade e movimentação do usuário, utilizando um algoritmo de classificação supervisionada. Com isso, torna-se possível a criação de aplicações para identificar uma queda em um idoso, permitindo ações de socorro mais rápidas, além de um cenário mais claro da distribuição de quedas, suas causas e contextos como subsídio para ações preventivas. Este tipo de estratégia já vem sendo aplicada para identificação de crises epilépticas [https://www.researchgate.net/profile/Khaled_Elleithy/publication/322921138_Smart_Phone_Application_Development_for_Monitoring_Epilepsy_Seizure_Detection_based_on_EEG_signal_Classification/links/5a76327145851541ce588920/Smart-Phone-Application-Development-for-Monitoring-Epilepsy-Seizure-Detection-based-on-EEG-signal-Classification.pdf?origin=publication_list, acesso em 09.07.2018].

### Descrição do problema

O problema a ser resolvido é um problema de classificação supervisionada. Dado uma observação ou um conjunto de observações provenientes de acelerômetros e sensores de um smartphone, necessitamos saber qual a atividade praticada pelo usuário daquele smartphone com algum grau de certeza. Podemos medir o quão bem resolvemos este problema a partir das taxas de acertos - numa situação de classificação binária (queda vs. não-queda), o número de falsos negativos não deveria ser um problema, e a métrica principal de revocação (recall), e não de precisão, torna-se mais apropriada. O problema é replicável, uma vez que o conjunto de dados encontra-se publicamente disponível.


### Conjuntos de dados e entradas

Serão utilizados os dados _Human Activity Recognition Using Smartphones Data Set_, disponíveis em [https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones]. Este conjunto foi produzido por um experimento no qual diversos voluntários praticaram diversas atividades com um aparelho preso à cintura - as informações geradas pelos acelerômetros foram preprocessadas e disponibilizadas. O conjunto de dados compreende cerca de 10000 observações de 561 caraterísticas (features), relacionadas a 7 atividades diferentes (`WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING`). Será tomada como atividade alvo a de deitar (`LAYING`), que pode corresponder a uma queda, e as demais serão agregadas, representado uma situação na qual não haja queda.

As características compreendem dados de acelerômetro e giroscópio em 3 direções. Os dados foram preprocessados de modo que, dentre as características, estao máximos, mínimos, médias e dados convertidos em termos de aceleração, aceleração gravitacional, energia - acredito que uma boa parte destes features podem estar de certo modo relacionados, podendo ser reduzidos.

Uma limitação do trabalho é o fato de o conjunto de dados ter sido gerado por voluntários entre 19 e 48 anos, cujo padrão motor provavelmente é bastante diferente do de idosos. Entende-se que o conjunto pode ser suficiente para o desenvolvimento do método e das estratégias de tratamento de dados e aprendizagem, mas que um novo conjunto de dados deveria ser gerado com uma amostra da população de interesse, e este conjunto utilizado para uma aplicação em produção.

Outra limitação é o conjunto de dados não conter informações de acelerometria diretamente medidas durante quedas ou simulações de quedas. Mais uma vez, os dados disponíveis permitirão desenvolver as bases metodológicas que poderão ser aprimoradas frente à maior disponibilidade de dados.


### Descrição da solução

A solução deverá ser um modelo de classificação binária supervisionada que permita identificar com algum grau de certeza quando um usuário está deitado, representando uma situação de queda (atividade `LAYING`), e quando ele está praticando outras atividades (não-queda).

Pensando que o modelo poderá ser aplicado em uma aplicação que monitore o estado do usuário em tempo real, permitindo comunicar qualquer ocorrência e tomada de providências, o classificador não pode ser exigente computacionalmente.

Ainda, uma situação de não-queda que seja apontada como queda (falso positivo) não deve ser relevante, podendo ser desconsiderada pelo usuário. Já um falso negativo é extremamente indesejado, pois poderá resultar no usuário em situação de queda, sem que haja predição desta situação e eventual notificação. Deste modo, a solução deverá ser otimizada para elevada revocação (recall), podendo ser mensurada utilizando este parâmetro.

A solução poderá ser reproduzida uma vez que parte do uso de software livre (Python e bibliotecas Pandas, Numpy e SKLearn), bem como dados abertos.


### Modelo de referência (benchmark)

Como incluído no histórico, aplicações semelhantes já foram realizadas tendo em vista o monitoramento e notificação de crises epilépticas. Porém, estas soluções valem-se de outros dados além dos de acelerometria em smartphones, como EEG por exemplo, sendo difícil a comparação com o que será desenvolvido aqui.

Além disso o conjunto de dados, por ser aberto, já foi utilizado em trabalhos de aprendizagem. Na plataforma Kaggle há algumas implantações de modelos de aprendizagem[https://www.kaggle.com/mboaglio/simplifiedhuarus/kernels, acesso em 10.07.2018], porém em todos trata-se de classificação multi-rótulos. Estas soluções, por serem implantadas em R, não poderão ser reproduzidas, mas poderão ter seu desempenho avaliado e comparado com o que será desenvolvido aqui.


### Métricas de avaliação

As principais métricas de avaliação adotadas serão:
- revocação (recall), isto é, a razão entre o total de positivos verdadeiros, e o total de ocorrências positivas.
- acurácia, isto é, razão entre total de acertos (positivos verdadeiros e negativos verdadeiros) e toda a população;

A primeira fornece uma visão específica da performance do modelo, ao passo que a segunda serve como uma visao geral, podendo servir como critério secundário de avaliação.


### Design do projeto

Toda a implementação será feita com software livre (Python 3.6 e bibliotecas Numpy, Pandas e SKLearn).

Inicialmente, será feita uma tratamento nos rótulos dos dados, de modo a reduzir de 6 (`WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING`) para 2 (`LAYING, NOT_LAYING`).

Em seguida, será análise exploratória rápida nos dados, para entendimento de correlações entre características que possam ser inicialmente desconsideradas, tendo em mente que muitas podem ter sido geradas em etapa de pretratamento. Poderá ser lançada mão análise de componentes principais para auxiliar nessa etapa, buscando reduzir o número de características de 561 para um número que seja significativamente menor. Imagino que 6 características podem dar conta de explicar os movimentos: 3 que representem movimentos lineares nas direções X, Y e Z, e 3 que representes movimentos angulares nos planos normais a estas direções. No entanto, o valor deverá depender de análise nos dados. A redução no número de características tem objetivo de simplificar o modelo, que pode ser vítima da 'maldição da dimensionalidade'.

O conjunto de dados será separado em um subconjunto de treino e um de teste, provavelmente numa razão de 0.7:0.3. O dado será randomizado antes da separação.

O conjunto de treino será utilizado para o modelo de aprendizagem. Serão testados os algoritmos de classificação: SVM, regressão logística, naive Bayes e árvore de decisão. O treino será realizado utilizando validação cruzada.

O algoritmo de melhor desempenho com o conjunto de teste deverá ter os hiperparâmetros ajustado com uma busca em matriz, que resultará no modelo final. Este será validado no conjunto de teste ao término do trabalho.

A cada etapa de avaliação serão reportados as métricas recall, acurácia, bem como uma matriz de confusão para melhor entendimento dos resultados.

Ao final do trabalho, será realizada uma discussão sobre a aplicabilidade do modelo, e eventuais oportunidades de melhoria.


-----------

**Antes de enviar sua proposta, pergunte-se. . .**

- A proposta que você escreveu segue uma estrutura bem organizada, similar ao modelo de projeto?
- Todas as seções (em especial, **Descrição da solução** e **Design do projeto**) estão escritas de uma forma clara, concisa e específica? Existe algum termo ou frase ambígua que precise de esclarecimento?
- O público-alvo de seu projeto será capaz de entender sua proposta?
- Você revisou sua proposta de projeto adequadamente, de forma a minimizar a quantidade de erros gramaticais e ortográficos?
- Todos os recursos usados neste projeto foram corretamente citados e referenciados?
