POWER_RELEASE = orion_power_beta_mar2003
CC = g++

CFLAGS =  -DS_OUT_DEBUG -Wno-deprecated  -g -DPOWER_TEST -I./$(POWER_RELEASE)/power -I./$(POWER_RELEASE)/library

LINKFLAGS = -L./$(POWER_RELEASE)/power -lm -lpower 
.SUFFIXES: .o .cc

HEADERS = index.h SString.h SStd.h SRGen.h Svector.h configuration.h mess_queue.h mess_event.h flit.h sim_foundation.h sim_router.h sim_vcarb.h

TARGET = popnet
PTARGET = power_model

SIM_SRCS = SString.cc SStd.cc SRGen.cc configuration.cc flit.cc sim_router.cc sim_routing.cc sim_vcarb.cc  mess_event.cc sim_foundation.cc mess_queue.cc main.cc

SIM_OBJS = $(SIM_SRCS:.cc=.o)

$(TARGET):$(SIM_OBJS) $(PTARGET) 
	$(CC) -o popnet $(SIM_OBJS) $(LINKFLAGS)

$(PTARGET):
	cd ./$(POWER_RELEASE)/power; $(MAKE)

configuration.o: configuration.cc configuration.h SStd.h SRGen.h index.h \
  SString.h
flit.o: flit.cc flit.h index.h SStd.h
main.o: main.cc index.h SStd.h SRGen.h configuration.h sim_foundation.h \
  sim_router.h flit.h mess_event.h mess_queue.h
mess_event.o: mess_event.cc mess_event.h index.h flit.h SStd.h
mess_queue.o: mess_queue.cc mess_queue.h SStd.h index.h configuration.h \
  SRGen.h sim_foundation.h sim_router.h flit.h mess_event.h
sim_foundation.o: sim_foundation.cc sim_foundation.h sim_router.h flit.h \
  index.h SStd.h SRGen.h configuration.h mess_event.h mess_queue.h 
sim_router.o: sim_router.cc sim_router.h flit.h index.h SStd.h SRGen.h \
  configuration.h sim_foundation.h mess_event.h mess_queue.h
sim_routing.o: sim_routing.cc sim_router.h flit.h index.h SStd.h SRGen.h \
  configuration.h sim_foundation.h mess_event.h mess_queue.h
sim_vcarb.o: sim_vcarb.cc sim_router.h flit.h index.h SStd.h SRGen.h \
  configuration.h sim_foundation.h mess_event.h mess_queue.h sim_vcarb.h
SRGen.o: SRGen.cc SRGen.h SStd.h
SStd.o: SStd.cc SStd.h
SString.o: SString.cc SString.h index.h SStd.h

.cc.o:
	$(CC) $(CFLAGS) -c $<  

clean:
	rm -rf *.o popnet core
	cd ./$(POWER_RELEASE)/power; $(MAKE) clean
