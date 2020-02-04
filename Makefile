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

gauss_deps_h 	= bcast_op
main_deps 		= fermi newtonroot findmu gauss 
main_deps_h 	= join bcast_op
findmu_deps 	= fermi newtonroot
dc_deps 		= TwoPara gauss
dc_deps_h 		= join bcast_op
TwoPara_deps 	= gauss dc
TwoPara_deps_h 	= join bcast_op arma_mpi_helper
FSSH_deps 		= dc fermi TwoPara gauss
FSSH_deps_h 	= join bcast_op

exe_test_src 	= $(addprefix ${BDIR}/test_, findmu fermi newtonroot gauss TwoPara dc FSSH interp)
exe_test_hdr 	= $(addprefix ${BDIR}/test_, join)
exe_all 		= ${BDIR}/main $(exe_test_src) $(exe_test_hdr)

obj_test_src 	= $(patsubst ${BDIR}/%, ${ODIR}/%.o, ${exe_test_src})
obj_test_hdr 	= $(patsubst ${BDIR}/%, ${ODIR}/%.o, ${exe_test_hdr})

gen_obj 		= $(patsubst %, ${ODIR}/%.o, $(1))
gen_hdr 		= $(patsubst %, ${IDIR}/%.h, $(1))


.PHONY: all
all 			: ${exe_all}
% 				: ${BDIR}/% ;


${BDIR}/main 	: ${ODIR}/main.o $(call gen_obj,${main_deps}) | ${BDIR}
	${CC} $^ -o $@ ${CPPFLAGS} ${LDFLAGS}

.SECONDEXPANSION:
${exe_test_src} : ${BDIR}/test_% : ${ODIR}/test_%.o ${ODIR}/%.o $$(call gen_obj,$${$$*_deps}) | ${BDIR}
	${CC} $^ -o $@ ${CPPFLAGS} ${LDFLAGS}

${exe_test_hdr} : ${BDIR}/test_% : ${ODIR}/test_%.o $$(call gen_obj,$${$$*_deps}) | ${BDIR}
	${CC} $^ -o $@ ${CPPFLAGS} ${LDFLAGS}


${ODIR}/main.o 	: main.cpp $(call gen_hdr,${main_deps_h} ${main_deps}) | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}

${obj_test_src} : ${ODIR}/test_%.o : test_%.cpp %.h $$(call gen_hdr,$${$$*_deps_h} $${$$*_deps}) | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}

${obj_test_hdr} : ${ODIR}/test_%.o : test_%.cpp %.h $$(call gen_hdr,$${$$*_deps_h} $${$$*_deps}) | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}

${ODIR}/%.o		: %.cpp %.h $$(call gen_hdr,$${$$*_deps_h} $${$$*_deps}) | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}


${BDIR} ${ODIR} ${LDIR} :
	mkdir -p $@


.PRECIOUS: ${ODIR}/%.o
.PHONY: clean

clean:
	rm -rf ${ODIR} ${BDIR} ${LDIR}

