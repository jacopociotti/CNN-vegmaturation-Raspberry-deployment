#ifndef CAMERA_HANDLER_H
#define CAMERA_HANDLER_H

#include <opencv2/opencv.hpp>
#include <functional>                                                   //<functional> serve per usare oggetti std::function, cioè funzioni, lambda o functor passabili come argomenti.

class CameraHandler {
public:                                                                 //i metodi e membri dichiarati qui sono accessibili dall’esterno della classe
    using FrameCallback = std::function<void(const cv::Mat&)>;          //FrameCallback è un oggetto funzione che prende in input una costante a un’immagine OpenCV (cv::Mat) e non ritorna nulla (void)
                                                                        //serve per passare una funzione callback da chiamare ad ogni frame catturato dalla webcam
    bool open(int deviceID = 0, int width = 640, int height = 480);     //Metodo pubblico open che apre la webcam. Restituisce true se l’apertura e la configurazione vanno a buon fine, false altrimenti.
    void loop(const FrameCallback& callback);                           //richiama callback ad ogni frame
    void close();

private:                                                                //Sezione privata: membri accessibili solo all’interno della classe.
    cv::VideoCapture cap;                                               //Oggetto OpenCV (capture) per gestire l’acquisizione video da webcam o file.
};
#endif

