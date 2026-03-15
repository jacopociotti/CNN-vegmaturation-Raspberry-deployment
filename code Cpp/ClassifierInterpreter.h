#ifndef CLASSIFIER_INTERPRETER_H
#define CLASSIFIER_INTERPRETER_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"

//Definizione della classe ClassifierInterpreter per l'interprete tflite
class ClassifierInterpreter {
public:
    explicit ClassifierInterpreter(const std::string& model_path);      //prende il percorso del modello .tflite e lo carica (explicit impedisce le conversioni implicite.)
    bool isValid() const;                                               //serve per verificare se il modello è stato caricato correttamente
    int infer(const cv::Mat& input_img, float& confidence);             //esegue l’inferenza sull’immagine input_img, restituisce l’indice della classe con maggior probabilità e scrive la confidenza in confidence

private:
    std::unique_ptr<tflite::FlatBufferModel> model;                     //model punta al modello TFLite caricato
    std::unique_ptr<tflite::impl::Interpreter> interpreter;             //oggetto che esegue l'inferenza
};

#endif


