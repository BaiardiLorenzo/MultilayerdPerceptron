# MultilayerdPerceptron
Implementazione del multilayerd perceptron multiclasse per problemi di classificazione. Per la riproduzione dei dati è necessario solamente far eseguire il programma dopo aver impostato correttamente le stringhe relative ai percorsi del dataset e, se si volesse cambiare i dataset, sarà sufficiente modificare le stringhe dei link all'interno dei metodo read_csv di pandas. I dataset dell'elabaroto sono stati presi dall'archivio: http://archive.ics.uci.edu/ml/datasets/Letter+RecognitionOpenCV-Python.

### Main
Modulo di Test. In questo modulo verrà chiamato l'algoritmo backpropagation per l'addestramento delle reti, attraverso due dataset. Successivamente verranno stampati i valori di error sul training e validation set.

### Active Function
Modulo nel quale sono presenti le varie funzioni di attivazioni con le relative derivate. Relu, Sigmoid, Tanh. E' presente anche la funzione di Softmax per il caso multiclasse.

### MLP
Classe che definisce la rete neurale MLP e il suo l'algoritmo di backpropagation.

### Init-Parameters
Modulo per l'inizializzazione dei parametri della rete. L'inizializzazione dei parametri può avvenire con Glorot (Distribuzione Uniforme) o con valori randomici.

### Graphics
Modulo dedicato alla stampa dei vari grafici.
