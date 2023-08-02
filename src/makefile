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

OBJ = $(SRC:%.cpp=%.o)

$(EXE): $(OBJ) libfftw3x_cdft_lp64.a
	$(CC) $(CFLAGS) $(INTELFLAGS) $(LIB) $^ -o $@ && cp -f $(EXE) ..

libfftw3x_cdft_lp64.a: fftw3x_cdft
	cd $< && $(MAKE) libintel64 MKLROOT=$(MKLROOT) INSTALL_DIR=".." && cd ..

fftw3x_cdft:
	cp -r $(MKLROOT)/interfaces/$@ .

%.o: %.cpp w3j.h
	$(CC) $(CFLAGS) $(OMP_FLAG) $(INC) -c $< -o $@

w3j.h:
	$(CC) to_w3j_h.cpp -o to_w3j_h && ./to_w3j_h

clean: 
	rm -f *.o */*.o to_w3j_h w3j.h $(EXE)

veryclean:
	rm -rf *.o */*.o ../$(EXE) $(EXE) *.a fftw3x_cdft obj_intel64_lp64 to_w3j_h w3j.h
