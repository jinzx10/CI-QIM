ROOTDIR = $(shell pwd)
SDIR = ${ROOTDIR}/src
IDIR = ${ROOTDIR}/include
ODIR = ${ROOTDIR}/obj
BDIR = ${ROOTDIR}/bin

vpath %.h ${IDIR}
vpath %.cpp ${SDIR}

CC 				= g++
CPPFLAGS 		= -I${IDIR}
LDFLAGS 		= -larmadillo

OBJ0 			= fermi 
OBJS 			= $(addsuffix .o, ${OBJ0})

main 			: ${OBJS} | ${BDIR}
	${CC} $(addprefix ${ODIR}/, $^) -o ${BDIR}/$@ ${CPPFLAGS} ${LDFLAGS}

test 			: test_fermi test_newtonroot
test_% 			: test_%.o %.o | ${BDIR}
	${CC} $(addprefix ${ODIR}/, $^) -o ${BDIR}/$@ ${CPPFLAGS} ${LDFLAGS}


main.o 			: main.cpp | ${ODIR}
	${CC} -c $< -o ${ODIR}/$@ ${CPPFLAGS}

test_%.o : test_%.cpp | ${ODIR}
	${CC} -c $< -o ${ODIR}/$@ ${CPPFLAGS}

%.o		: %.cpp %.h | ${ODIR}
	${CC} -c $< -o ${ODIR}/$@ ${CPPFLAGS}

${BDIR} ${ODIR} ${LDIR} :
	mkdir -p $@

.PRECIOUS: %.o test_%.o
.PHONY: clean

clean:
	rm -rf ${ODIR} ${BDIR} ${LDIR}

