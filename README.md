# 🍅 Classificazione di Ortaggi e Stato di Maturazione su Raspberry Pi (Cortex-A)

Questo progetto implementa un sistema basato su Reti Neurali Convoluzionali (CNN) custom per riconoscere in tempo reale il tipo di ortaggio e determinarne lo stato di maturazione. I modelli sono ottimizzati ed eseguiti su un sistema embedded (Raspberry Pi 3 Model B con processore ARM Cortex-A) acquisendo i frame video tramite una webcam USB.

L'applicativo software in ambiente di produzione è sviluppato interamente in **C++**, avvalendosi di OpenCV per la computer vision e TensorFlow Lite C++ API per l'inferenza.

## 📊 Dataset e Data Augmentation
Il progetto sfrutta il *Vegetable Classification Dataset* reperibile su Mendeley Data. 
* **Classi principali (Tipi di Ortaggi)**: Bell Pepper, Chile Pepper, New Mexico Green Chile, Tomato.
* **Sottoclassi (Stati di Maturazione)**: Damaged, Dried, Old, Ripe, Unripe.

Per via di un forte sbilanciamento iniziale, è stata applicata una pipeline di **Data Augmentation** (rotazioni, zoom, riflessioni) per bilanciare le classi fino a raggiungere circa 3000 immagini per classe principale e 600 per ogni sottoclasse. Il dataset bilanciato è stato poi diviso in Train (70%), Validation (15%) e Test (15%).

## 🧠 Architettura dei Modelli (Python / TensorFlow)
Il sistema è composto da una gerarchia di **5 modelli CNN custom**:
1. **Vegetable Classifier**: Un modello primario che predice il tipo di ortaggio.
2. **Maturation Classifiers**: Quattro modelli specifici (uno per ogni tipo di ortaggio) chiamati a cascata per predirne l'esatto stato di maturazione in base al primo risultato.

Tutte le reti presentano layer `Conv2D`, `BatchNormalization`, `MaxPooling2D`, livelli `Dense` e `Dropout` per prevenire l'overfitting.
Per consentire l'esecuzione sul dispositivo embedded, i modelli Python (`.h5`) sono stati convertiti nel formato **TensorFlow Lite (`.tflite`)** applicando una quantizzazione *float16*, riducendo drasticamente il peso (da ~127 MB a ~21 MB ciascuno).

## 💻 Software Architecture in C++
L'inferenza sul dispositivo embedded è gestita da un applicativo C++ diviso in tre componenti:
* **`CameraHandler`**: Modulo asincrono basato su OpenCV per la cattura continua dei frame video dalla webcam USB tramite funzioni callback.
* **`ClassifierInterpreter`**: Classe wrapper per le API C++ di TFLite che si occupa di inizializzare il modello FlatBuffer, pre-processare le matrici (`cv::Mat` in input) ed estrarre la label con la confidenza maggiore tramite la funzione `Invoke()`.
* **`main`**: Coordina il flusso, passando prima l'immagine al modello principale e poi innescando dinamicamente il modello di maturazione corretto per stampare i risultati a schermo in tempo reale.

## 🛠️ Cross-Compilazione per ARMv7
Per evitare i lunghissimi tempi di compilazione a bordo della Raspberry Pi, l'intero eseguibile è stato cross-compilato su una macchina host Ubuntu (x86) per architettura ARM 32-bit.
* **Toolchain**: GNU Arm Embedded Toolchain (`arm-none-linux-gnueabihf-g++`).
* Le dipendenze (TensorFlow Lite, FlatBuffers, Abseil, OpenCV) sono state compilate staticamente (`.a`) e linkate via CMake/Makefile per generare un singolo file eseguibile `.elf` standalone.

## 🚀 Esecuzione
L'eseguibile compilato (`classifier.elf`) viene trasferito sul Raspberry Pi tramite protocollo SSH/SCP.

Per avviare l'inferenza in tempo reale con la webcam collegata, eseguire dal terminale della Raspberry:
`./classifier`

Il programma stamperà in console le predizioni continue ad ogni frame, indicando l'ortaggio rilevato, lo stato di maturazione e la relativa percentuale di confidenza.
