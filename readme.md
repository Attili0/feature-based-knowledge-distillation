# Feature-Based Knowledge Distillation

**Estudo Comparativo de Esquemas de Destilação de Conhecimento Baseados em Características Latentes de Classificadores Pré-treinados**

Este repositório contém o código e a fundamentação teórica para o projeto final da disciplina MO434 do Instituto de Computação da Universidade Estadual de Campinas. O trabalho investiga a eficácia de abordagens de Destilação de Conhecimento Baseada em Características aplicadas a redes neurais convolucionais leves voltadas para dispositivos de borda.

O objetivo central consiste em forçar arquiteturas compactas a mimetizarem as representações latentes geradas logo após a camada de agrupamento espacial global (post-GAP) de classificadores de grande porte pré-treinados no ImageNet.

## Modelos Avaliados

* **Modelos Professores**: VGG (11, 16 e 19 com Batch Normalization), ResNet (18, 50 e 101\) e ConvNeXt (Tiny, Small, Base e Large).  
* **Modelos Estudantes**: MobileNet V2 (com fator de largura comprimido, `width_mult = 0.5`) e ShuffleNet V2.

## Metodologia e Pipeline

O pipeline experimental foi desenvolvido para execução paralela e assíncrona utilizando uma arquitetura baseada em filas distribuídas entre múltiplos aceleradores de hardware. O processo divide-se em duas fases:

1. **Fase 1: Sondagem Linear (Linear Probing)**  
   * Os blocos extratores dos professores foram totalmente congelados.  
   * O treinamento focou apenas no ajuste de uma camada linear de classificação conectada à saída da camada GAP, garantindo a avaliação de projeções puras do conhecimento original do ImageNet.  
2. **Fase 2: Destilação Baseada em Características**  
   * O classificador padrão das redes estudantes foi substituído por uma operação de identidade.  
   * A destilação operou via regressão por Erro Quadrático Médio (MSE), assistida por um módulo preditor denso multicamadas (MLP Predictor) para compatibilizar a dimensionalidade dos vetores do estudante e do professor.  
   * A função de perda otimizada combinou o MSE latente e a perda de entropia cruzada obtida pelo classificador já treinado do professor.

### Parâmetros de Treinamento

* **Datasets**: CIFAR-100 e Intel Scenes.  
* **Resolução**: Imagens redimensionadas e normalizadas uniformemente para 224x224 pixels.  
* **Batch Size**: Fixado em 512 amostras para amortecer flutuações e ruídos nas estimativas de gradiente.

## Principais Resultados

* **O Paradoxo do Professor Perfeito**: O aumento da capacidade absoluta do professor não resultou em um estudante melhor. Professores complexos, como os da família ConvNeXt, geraram descompasso de capacidade (capacity mismatch) e estrangulamento de posto (rank bottleneck) ao projetar espaços hiper-dimensionais complexos demais para as redes leves aprenderem.  
* **A Eficiência da ResNet-18**: A arquitetura ResNet-18 provou ser a professora mais eficiente. Por projetar um espaço latente de dimensionalidade moderada (512 dimensões) e menor variância geométrica, ofereceu um alvo alcançável, mitigando o colapso de aprendizado dos estudantes.  
* **Alinhamento Arquitetural**: A MobileNet V2 estabeleceu acurácias consistentemente superiores à ShuffleNet V2. Isso decorre de sua topologia baseada em blocos residuais invertidos, que possui forte alinhamento e afinidade matemática com as ResNets e ConvNeXts.  
* **Anomalia da VGG-19**: A destilação utilizando a VGG-19 gerou um salto abrupto e anômalo na acurácia dos estudantes no dataset Intel Scenes, indicando que a contração de suas representações gerou um espaço latente geometricamente favorável para tarefas de baixa complexidade.

## Propostas para Trabalhos Futuros

Com base nas limitações empíricas observadas, sugerem-se as seguintes melhorias na arquitetura e modelagem:

* **Granularidade Espacial**: Evoluir a destilação post-GAP para o uso de mapas de características pre-GAP, substituindo projetores lineares por blocos convolucionais adaptativos para preservar a semântica espacial.  
* **Aprendizado Contrastivo**: Substituir o alinhamento puramente geométrico (MSE) por métodos de Destilação Contrastiva (CRD) para evitar o gargalo destrutivo causado pela restrição de pareamento um-para-um, permitindo ao modelo formar seu próprio manifold.  
* **Destilação Relacional (RKD)**: Focar na minimização da divergência das relações métricas e angulares entre as instâncias do lote em vez de mapear coordenadas latentes absolutas.  
* **Reformulação do Projetor**: Evitar o estrangulamento de posto expandindo dinamicamente a dimensionalidade da camada oculta do MLP e adotar uma abordagem de Conjunto de Projetores (Projector Ensemble) para alargar o fluxo de conhecimento.

## Autor

* **Rafael Attilio Agricola** \- Universidade Estadual de Campinas (UNICAMP)
