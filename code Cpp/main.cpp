#include <iostream>
#include <map>
#include <vector>
#include <thread>
#include <chrono>
#include "CameraHandler.h"
#include "ClassifierInterpreter.h"

//Definizione classi principali (tipo di ortaggio)
const std::vector<std::string> fruit_classes = {
    "1. Bell Pepper", "2. Chile Pepper", "3. New Mexico Green Chile", "4. Tomato"
};

//Definizione sottoclassi (stato di maturazione)
const std::vector<std::string> maturity_classes = {
    "Damaged", "Dried", "Old", "Ripe", "Unripe"
};

//Mappa da tipo di ortaggio al path del modello specifico per lo stato di maturazione
const std::map<std::string, std::string> maturity_model_paths = {
    {"1. Bell Pepper", "/home/jacop/models/maturity_bellpepper4.tflite"},
    {"2. Chile Pepper", "/home/jacop/models/maturity_chilepepper4.tflite"},
    {"3. New Mexico Green Chile", "/home/jacop/models/maturity_newmexchilepepper4.tflite"},
    {"4. Tomato", "/home/jacop/models/maturity_tomato4.tflite"}
};

//main
int main() {
    ClassifierInterpreter fruit_model("/home/jacop/models/fruit_type_classifier4.tflite");      //Crea un interprete TFLite per il modello che classifica il tipo di ortaggio

    if (!fruit_model.isValid()) {
        std::cerr << "Modello tipo frutto non valido\n";
        return 1;
    }

    CameraHandler camera;       //Crea il gestore della webcam.
    if (!camera.open()) {
        return 1;
    }

    std::cout << "Premi Ctrl+C per terminare\n";

    //Avvia il loop di acquisizione webcam. Per ogni frame acquisito, viene chiamata la funzione lambda callback
    // [capture](parametri){        "capture se per valore o per riferimento(&)"
    // }.
    //[&] indica che la funzione callback cattura per riferimento tutte le variabili esterne usate (ad es. fruit_model).
    camera.loop([&](const cv::Mat& frame) {

        float conf_fruit;
        int fruit_idx = fruit_model.infer(frame, conf_fruit);       //esegue inferenza sul frame con il primo modello (tipo di ortaggio) e riceve indice della classe predetta (confidenza)

        // Check validità
        if (fruit_idx < 0 || fruit_idx >= fruit_classes.size()) {
            std::cerr << "Indice frutto non valido: " << fruit_idx << "\n";
            return;
        }


        std::string fruit_label = fruit_classes[fruit_idx];         //ricava il nome dell'ortaggio dall'indice
        std::cout << "Frutto rilevato: " << fruit_label << "\n";

        auto it = maturity_model_paths.find(fruit_label);           //cerca nella mappa il modello dello stato di maturazione corrispondente all'ortaggio
        if (it == maturity_model_paths.end()) {
            std::cerr << "Nessun modello di maturazione per: " << fruit_label << "\n";
            return;
        }


        ClassifierInterpreter maturity_model(maturity_model_paths.at(fruit_label));   //Crea un interprete TFLite per il modello che classifica lo stato di maturazione

        float conf_maturity;
        int maturity_idx = maturity_model.infer(frame, conf_maturity);                //esegue inferenza sul frame con il modello (stato di maturazione) e riceve indice della classe predetta (confidenza)

        //Stampa i risultati
        std::cout << "Frutto: " << fruit_label << " (" << conf_fruit << ")\n";
        std::cout << "Maturazione: " << maturity_classes[maturity_idx] << " (" << conf_maturity << ")\n";
        std::cout << "-----------------------------\n";

        //Delay tra le inferenze
        //std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    });

    return 0;
}


