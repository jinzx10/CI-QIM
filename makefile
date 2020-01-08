ROOTDIR = $(shell pwd)
SDIR = ${ROOTDIR}/src
IDIR = ${ROOTDIR}/include

ODIR = ${ROOTDIR}/obj
LDIR = ${ROOTDIR}/lib
BDIR = ${ROOTDIR}/bin

vpath %.h ${IDIR}
vpath %.cpp ${SDIR}

CC = g++
CPPFLAGS = -I${IDIR}

OBJ0 = 
STATIC_LIB0 = 
SHARED_LIB0 = 

LDFLAGS = -L${LDIR} -Wl,-rpath=${LDIR}
LDFLAGS += $(foreach LIB, ${STATIC_LIB0} ${SHARED_LIB0}, -l${LIB})
LDFLAGS += -larmadillo

OBJS = $(patsubst %, ${ODIR}/%.o, ${OBJ0})
STATIC_LIBS = $(patsubst %, ${LDIR}/lib%.a, ${STATIC_LIB0})
SHARED_LIBS = $(patsubst %, ${LDIR}/lib%.so, ${SHARED_LIB0})


main 			: ${OBJS} ${STATIC_LIBS} ${SHARED_LIBS} | ${BDIR}
	${CC} ${OBJS} -o ${BDIR}/$@ ${CPPFLAGS} ${LDFLAGS}

test_fermi 		: ${ODIR}/test_fermi.o ${ODIR}/fermi.o | ${BDIR}
	${CC} $^ -o ${BDIR}/$@ ${CPPFLAGS} ${LDFLAGS}

${ODIR}/main.o 	: main.cpp | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}

${ODIR}/test_fermi.o : test_fermi.cpp | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}


${ODIR}/%.o		: %.cpp %.h | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}

${LDIR}/lib%.a  : ${ODIR}/%.o | ${LDIR}
	ar -rcs $@ $<

${LDIR}/lib%.so : ${ODIR}/%.o | ${LDIR}
	${CC} -shared -o $@ $<

${BDIR} ${ODIR} ${LDIR} :
	mkdir -p $@


.PHONY: clean
clean:
	rm -rf ${ODIR} ${BDIR} ${LDIR}

