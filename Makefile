MPICXX ?= mpic++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
OPENMP_FLAGS ?= -fopenmp
PTHREAD_FLAGS ?= -pthread

COMMON_SRC = train.cpp guessing.cpp md5.cpp
GENERAL_SRC = correctness_guess_general.cpp $(COMMON_SRC)

.PHONY: all clean general no_openmp

all: general

general: correctness_guess_general

correctness_guess_general: $(GENERAL_SRC) PCFG.h md5.h
	$(MPICXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(PTHREAD_FLAGS) -o $@ $(GENERAL_SRC)

# 如果某台机器没有 OpenMP，可以用 make no_openmp。
# 此时 --local-mode openmp 会自动退化为串行生成。
no_openmp: $(GENERAL_SRC) PCFG.h md5.h
	$(MPICXX) $(CXXFLAGS) $(PTHREAD_FLAGS) -o correctness_guess_general $(GENERAL_SRC)

clean:
	rm -f correctness_guess_general *.o
