
APLT_ROOT?=~/gits/apltail
APLT?=$(APLT_ROOT)/aplt
PRELUDE?=$(APLT_ROOT)/lib/prelude.apl
TAIL2FUT?=tail2futhark
FUTHARK_C?=futhark-c

.PHONY: all
all: network_gen.out
	cat $<

.PHONY: interp
interp:
	$(APLT) -s_tail -p_tail $(PRELUDE) network.apl

%.tl: %.apl Makefile
	$(APLT) -p_types -s_tail -c -o $@ $(PRELUDE) $<

%_gen.fut: %.tl Makefile
	$(TAIL2FUT) -o $@ $<

%_gen.exe: %_gen.fut
	$(FUTHARK_C) -o $@ $<

%_gen.out: %_gen.exe
	cat ../futhark/mnist_all.fut.txt | ./$< > $@

clean:
	rm -rf *~ *_gen.fut *.tl *_gen.exe *_gen.tl *_gen.c
