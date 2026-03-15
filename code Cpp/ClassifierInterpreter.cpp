#include "ClassifierInterpreter.h"
#include <algorithm>        //<algorithm> per funzioni STL come std::max_element.
#include <cstring>          //<cstring> per std::memcpy.

//Carica il modello TFLite da file
ClassifierInterpreter::ClassifierInterpreter(const std::string& model_path) {
    model = tflite::impl::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) return;

    tflite::ops::builtin::BuiltinOpResolver resolver;           //crea un BuiltinOpResolver per registrare operatori standard TFLite.
    tflite::impl::InterpreterBuilder builder(*model, resolver); //Costruisce l’interprete associato al modello.

    builder(&interpreter);
    if (interpreter) interpreter->AllocateTensors();            //alloca memoria per input/output.
}

//Ritorna true se l’interprete è stato correttamente creato (modello valido).
bool ClassifierInterpreter::isValid() const {
    return interpreter != nullptr;
}

//Definizione metodo inferenza
int ClassifierInterpreter::infer(const cv::Mat& input_img, float& confidence) {
    if (!interpreter) return -1;

    int input = interpreter->inputs()[0];       //prende l’indice del tensore di input (di solito 0)

    if (input < 0) {
        std::cerr << "Indice input non valido\n";
        return -1;
    }

    TfLiteIntArray* dims = interpreter->tensor(input)->dims;    //recupera la forma (dimensioni) del tensore di input.

    //Verifica che sia 4D (batch, altezza, larghezza, canali).
    if (!dims || dims->size != 4) {
        std::cerr << "Dimensioni input non valide\n";
        return -1;
    }


    int height = dims->data[1];
    int width  = dims->data[2];
    int channels = dims->data[3];

    if (input_img.empty()) {
        std::cerr << "Immagine vuota\n";
        return -1;
    }



    //Converte da BGR a RGB, ridimensiona immagine a dimensione input modello e converte tipo a float32 con valori normalizzati da 0 a 1.
    cv::Mat rgb;
    cv::cvtColor(input_img, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(width, height));
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

    float* input_tensor = interpreter->typed_input_tensor<float>(0);        //recupera il puntatore all'input
    std::memcpy(input_tensor, resized.data, sizeof(float) * width * height * channels); //copia i dati normalizzati nella memoria di input del tensore TFLite.

    //Esegue l'inferenza
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Fallita l'inferenza\n";
        return -1;
    }

    const float* output = interpreter->typed_output_tensor<float>(0);   //recupera il puntatore all’output (probabilità delle classi).

    if (!output) {
        std::cerr << "Accesso a output_tensor fallito\n";
        return -1;
    }

    int output_size = interpreter->tensor(interpreter->outputs()[0])->bytes / sizeof(float);        //calcola la dimensione (numero di classi) dell’output.

    if (output_size <= 0) {
        std::cerr << "Output tensor vuoto\n";
        return -1;
    }

    //max_element trova il puntatore al valore massimo.
    //distance converte quel puntatore in un indice intero rispetto all’inizio dell’array.
    int best_idx = std::distance(output, std::max_element(output, output + output_size));   //trova l’indice della classe con la massima probabilità (argmax). output è un puntatore al primo elemento di un array di float,
                                                                                            //cioè l’inizio del vettore di output (ad esempio probabilità delle classi).
                                                                                            //output + output_size è un puntatore che indica la posizione dopo l’ultimo elemento del vettore
                                                                                            //std::max_element(output, output + output_size) è una funzione della libreria <algorithm>.
                                                                                            //Cerca e ritorna un puntatore all’elemento con valore massimo nel range indicato.
                                                                                            //ritonra quindi un puntatore all’elemento con valore più alto nel vettore output.
                                                                                            //std::distance() calcola la distanza (numero di elementi) fra due puntatori, misura la distanza fra output (inizio) e il puntatore ritornato da
                                                                                            //max_element. Calcola l’indice dell’elemento massimo nel vettore

    confidence = output[best_idx];      //assegna la confidenza relativa.
    return best_idx;                    //ritorna l’indice della classe predetta.
}

