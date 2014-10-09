LIB = -L/usr/local/include/
LINK = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_nonfree -lopencv_features2d -lopencv_objdetect 
TARGS = main.o  hogBlob.o sobel.o canny.o harris.o harrisLaplace.o
a.out: $(TARGS)
	g++ -g $(LINK) $(TARGS) 
main.o: imgFeat.h main.cpp
	g++ -c -g $(LIB) main.cpp
hogBlob.o: imgFeat.h hogBlob.cpp
	g++ -c -g $(LIB) hogBlob.cpp
soble.o: imgFeat.h sobel.cpp
	g++ -c -g $(LIB) sobel.cpp
canny.o: imgFeat.h canny.cpp
	g++ -c -g $(LIB) canny.cpp
harris.o: imgFeat.h harris.cpp
	g++ -c -g $(LIB) harris.cpp
harrisLaplace.o:imgFeat.h harrisLaplace.cpp
	g++ -c -g $(LIB) harrisLaplace.cpp
clean:
	rm main $(TARGS)
