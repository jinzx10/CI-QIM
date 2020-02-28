ROOTDIR 		= $(shell pwd)
BDIR 			= ${ROOTDIR}/bin
IDIR 			= ${ROOTDIR}/include
ODIR 			= ${ROOTDIR}/obj
SDIR 			= ${ROOTDIR}/src

CC 				= mpicxx
CPPFLAGS 		= -std=c++11 -O2 -I${IDIR} -DARMA_ALLOW_FAKE_GCC # -DNO_VELOCITY_REVERSAL
LDFLAGS 		= -larmadillo

vpath %.h ${IDIR}
vpath %.cpp ${SDIR}

TwoPara_deps 			= 
TwoPara_deps_h 			= arma_helper math_helper 

TwoPara2_deps 			= 
TwoPara2_deps_h 		= math_helper

FSSH_deps 				= TwoPara2
FSSH_deps_h 			= arma_helper math_helper

Langevin_deps 			= TwoPara2
Langevin_deps_h 		= arma_helper math_helper

SIAM_deps 				=
SIAM_deps_h 			= arma_helper math_helper 

executables	 	= $(addprefix ${BDIR}/run_, TwoPara TwoPara2 FSSH Langevin SIAM)
obj_run 		= $(patsubst ${BDIR}/%, ${ODIR}/%.o, ${executables})
gen_obj 		= $(patsubst %, ${ODIR}/%.o, $(1))
gen_hdr 		= $(patsubst %, ${IDIR}/%.h, $(1))


.PHONY: all
all 			: ${executables}
% 				: ${BDIR}/% ;

.SECONDEXPANSION:
${executables} 	: ${BDIR}/run_% : ${ODIR}/run_%.o ${ODIR}/%.o $$(call gen_obj,$${$$*_deps}) | ${BDIR}
	${CC} $^ -o $@ ${CPPFLAGS} ${LDFLAGS}

${obj_run} : ${ODIR}/run_%.o : run_%.cpp %.h $$(call gen_hdr,$${$$*_deps_h}) arma_helper.h mpi_helper.h widgets.h | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}

${ODIR}/%.o		: %.cpp %.h $$(call gen_hdr,$${$$*_deps_h}) | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}


${BDIR} ${ODIR} :
	mkdir -p $@


.PRECIOUS: ${ODIR}/%.o
.PHONY: clean

clean:
	rm -rf ${ODIR} ${BDIR} ${LDIR}

