EXE = HFNAMD
CC = mpiicpc

CFLAGS = -g -O2 -std=c++20
OMP_FLAG = -qopenmp
INC = -I $(MKLROOT)/include \
      -I $(MKLROOT)/include/fftw
LIB = -L $(MKLROOT)/lib/intel64
#INTELFLAGS = -mkl \
	     -lmkl_scalapack_lp64 \
	     -lmkl_intel_lp64 \
	     -lmkl_intel_thread \
	     -lmkl_core \
	     -lmkl_blacs_intelmpi_lp64 \
	     -liomp5 -lpthread -ldl 
INTELFLAGS = -lmkl_scalapack_lp64 -lmkl_cdft_core -lmkl_blacs_intelmpi_lp64 -liomp5 -lpthread -ldl -mkl


SRC =        fn.cpp \
           math.cpp \
            sym.cpp \
            paw.cpp \
      wave_base.cpp \
	   wave.cpp \
	    soc.cpp \
             io.cpp \
	    nac.cpp \
	   wpot.cpp \
	    bse.cpp \
	optical.cpp \
       dynamics.cpp \
	  tdcft.cpp \
	    dsh.cpp \
        hopping.cpp \
           main.cpp

OBJ = $(addprefix src/, $(SRC:%.cpp=%.o))

$(EXE): $(OBJ) src/libfftw3x_cdft_lp64.a
	cd ./src && $(CC) $(CFLAGS) $(INTELFLAGS) $(LIB) $(^F) -o $@ && cp -f $(EXE) .. && cd ..

src/libfftw3x_cdft_lp64.a: src/fftw3x_cdft
	cd $< && $(MAKE) libintel64 MKLROOT=$(MKLROOT) INSTALL_DIR=".." && cd ../..

src/fftw3x_cdft:
	cp -r $(MKLROOT)/interfaces/fftw3x_cdft ./src/

src/%.o: src/%.cpp src/w3j.h
	cd ./src && $(CC) $(CFLAGS) $(OMP_FLAG) $(INC) -c $(<F) -o $(@F)

src/w3j.h:
	cd ./src && $(CC) to_w3j_h.cpp -o to_w3j_h && ./to_w3j_h && cd ..

clean: 
	rm -f src/*.o src/*/*.o src/to_w3j_h src/w3j.h $(EXE) src/$(EXE)

veryclean:
	rm -rf src/*.o src/*/*.o $(EXE) src/$(EXE) src/*.a src/fftw3x_cdft src/obj_intel64_lp64 src/to_w3j_h src/w3j.h
