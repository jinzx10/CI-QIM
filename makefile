ROOTDIR 		= $(shell pwd)
BDIR 			= ${ROOTDIR}/bin
IDIR 			= ${ROOTDIR}/include
ODIR 			= ${ROOTDIR}/obj
SDIR 			= ${ROOTDIR}/src

CC 				= g++
CPPFLAGS 		= -I${IDIR}
LDFLAGS 		= -larmadillo

vpath %.h ${IDIR}
vpath %.cpp ${SDIR}

exe 			= $(addprefix test_, findmu fermi newtonroot)

findmu_deps 	= fermi newtonroot

.PHONY: all
all 			: $(addprefix ${BDIR}/, ${exe})
% 				: ${BDIR}/% ;

exe_objs 		= $(patsubst %, ${ODIR}/%.o, ${exe})
genobj 			= ${ODIR}/$(1).o $(patsubst %, ${ODIR}/%.o, $(2)) 

.SECONDEXPANSION:
${BDIR}/test_%	: ${ODIR}/test_%.o $$(call genobj,%,$$($$*_deps)) | ${BDIR}
	${CC} $^ -o $@ ${CPPFLAGS} ${LDFLAGS}

${exe_objs} 	: ${ODIR}/%.o : %.cpp | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}

${ODIR}/%.o		: %.cpp %.h | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}

${BDIR} ${ODIR} ${LDIR} :
	mkdir -p $@


.PRECIOUS: ${ODIR}/%.o
.PHONY: clean

clean:
	rm -rf ${ODIR} ${BDIR} ${LDIR}

