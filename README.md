# MultilayerdPerceptron
Implementazione del multilayerd perceptron multiclasse per problemi di classificazione in python. Per la riproduzione dei dati è necessario far eseguire il programma. Se si ha intenzione di cambiare i parametri dei layer nascosti è sufficiente modificare la lista h_layers nel main. Per cambiare i dataset forniti dal sito: http://archive.ics.uci.edu/ml/datasets/Letter+RecognitionOpenCV-Python è sufficiente modificare la stringa di link all'interno del metodo read_csv di pandas.

### Main
Modulo di Test. In questo modulo verrà chiamato l'algoritmo backpropagation per l'addestramento della reta, attraverso due dataset. Successivamente verranno stampati i valori di error sul training e validation set.

### Active Function
Modulo nel quale sono presenti le varie funzioni di attivazioni con le relative derivate. In particolare: Relu, Sigmoid.

### MLP
Classe che definisce la rete neurale MLP. E' presente l'algoritmo di backpropagation con relativo calcolo del gradiente.

### Softmax
Modulo contenente la funzione di softmax per la multiclasse.

### Init-Weight
Modulo per l'inizializzazione dei pesi della MLP. E' presente l'inizializzazione Uniforme di Glorot e randomica.

### Graphics
Modulo dedicato alla stampa e alla visualizzazione della rete.
