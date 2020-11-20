Grandes blocos de codigo faltante:
- Implantar classificadores texto NN:
    - ~~representacao Word2vec~~
    - ~~classificador CNN~~
    - ~~representacao LSTM (Cho et al)~~
    - classificador ELMO (ver AllenNLP, tem portugues pre treinado no site)
- Adicionar outros algoritmos de representacao de grafos:
    - ~~LLE~~
    - ~~GCN (usar o grau como feature?)~~
    - Learning Structural Node Embeddings via Diffusion Wavelets (tem na lib stellar, parecido com node2vec, precisaria escrever no texto)
    - ~~investigar Deep Graph Infomax~~
- Juntar embedding texto + embedding grafo no classificador
    - usar eles pre-treinados individualmente e fazer fine-tunning?

- Criar um data/selected_users.csv com os usuarios ordenados por prioridade na selecao de grafo
- tambem ja definir um data/test_users.csv entre os outros arquivos de supervisao distante separados por teste/validacao dos algoritmos com grafos
