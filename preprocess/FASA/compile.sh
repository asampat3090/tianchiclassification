if [ "$(uname)" == "Darwin" ]; then
    # Do something under Mac OS X platform
     g++ FASA.cpp -I/usr/local/include -L/usr/local/lib -lopencv_highgui -lopencv_core -lopencv_imgproc -o FASA
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Do something under GNU/Linux platform
     g++ -std=c++0x FASA.cpp -I/usr/local/include -L/usr/local/lib -lopencv_imgcodecs -lopencv_core -lopencv_imgproc -lopencv_videoio -o FASA
fi