#include "CameraHandler.h"
#include <iostream>                                                 //iostream per stampare messaggi di debug o errori sulla console
#include <chrono>                                                   //chrono e thread per gestire pause e sleep nel loop di acquisizione
#include <thread>

//Definizione del metodo open della classe CameraHandler-serve ad aprire la webcam
bool CameraHandler::open(int deviceID, int width, int height) {
    cap.open(deviceID);                                             //oggetto cap (di tipo cv::VideoCapture) per aprire la webcam specificata con il backend V4L2 (Video4Linux2)
    //controllo se la cam è aperta
    if (!cap.isOpened()) {
        std::cerr << "Impossibile aprire la webcam\n";
        return false;
    }

    //imposto la risoluzione della webcam
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    return true;
}

//Definizione del metodo loop
//Avvia un ciclo continuo di acquisizione immagini dalla webcam.
//Per ogni frame catturato, chiama la funzione callback passata come parametro.
//callback è una funzione che riceve un const cv::Mat& (frame immagine).
void CameraHandler::loop(const FrameCallback& callback) {
    while (true) {
        cv::Mat frame;                  //oggetto cv::Mat chiamato frame che conterrà l’immagine acquisita.
        cap >> frame;                   //operatore >> per leggere un frame dalla webcam cap e inserisce in frame.
        //Se il frame non è vuoto
        if (!frame.empty()) {
            callback(frame);            //chiama la funzione callback fornendo il frame letto. In questo modo il chiamante può elaborare il frame
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));  // delay di 200 ms
    }
}

//Definizione metodo close che chiude la webcam se è aperta.
void CameraHandler::close() {
    if (cap.isOpened())
        cap.release();
}




