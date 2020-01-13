ROOTDIR 		= $(shell pwd)
BDIR 			= ${ROOTDIR}/bin
IDIR 			= ${ROOTDIR}/include
ODIR 			= ${ROOTDIR}/obj
SDIR 			= ${ROOTDIR}/src

CC 				= mpicxx
CPPFLAGS 		= -I${IDIR} -O2 
LDFLAGS 		= -larmadillo

vpath %.h ${IDIR}
vpath %.cpp ${SDIR}

main_deps 		= fermi newtonroot findmu gauss 
findmu_deps 	= fermi newtonroot
TwoPara_deps 	= gauss
TwoPara_deps_h 	= join

exe 			= main $(addprefix test_, findmu fermi newtonroot gauss TwoPara join_)
exe_objs 		= $(patsubst %, ${ODIR}/%.o, ${exe})

gen_obj 		= $(patsubst %, ${ODIR}/%.o, $(2))
gen_hdr 		= $(patsubst %, ${IDIR}/%.h, $(2))

.PHONY: all
all 			: $(addprefix ${BDIR}/, ${exe})
% 				: ${BDIR}/% ;

${BDIR}/main 	: $(call gen_obj,main,${main_deps}) | ${BDIR}
	${CC} $^ -o $@ ${CPPFLAGS} ${LDFLAGS}

.SECONDEXPANSION:
${BDIR}/test_%	: ${ODIR}/test_%.o ${ODIR}/%.o $$(call gen_obj,$${$$*_deps}) | ${BDIR}
	${CC} $^ -o $@ ${CPPFLAGS} ${LDFLAGS}

${BDIR}/test_%_	: ${ODIR}/test_%.o $$(call gen_obj,$${$$*_deps}) | ${BDIR}
	${CC} $^ -o $@ ${CPPFLAGS} ${LDFLAGS}

${exe_objs} 	: ${ODIR}/%.o : %.cpp | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}

${ODIR}/%.o		: %.cpp %.h $$(call gen_hdr,$${$$*_deps_h}) | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}

${BDIR} ${ODIR} ${LDIR} :
	mkdir -p $@


.PRECIOUS: ${ODIR}/%.o
.PHONY: clean

clean:
	rm -rf ${ODIR} ${BDIR} ${LDIR}

