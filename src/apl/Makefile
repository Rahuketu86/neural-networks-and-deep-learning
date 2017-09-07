
APLT_ROOT?=~/gits/apltail
APLT?=$(APLT_ROOT)/aplt
PRELUDE?=$(APLT_ROOT)/lib/prelude.apl
TAIL2FUT?=tail2futhark

.PHONY: all
all: network.tl

.PHONY: interp
interp:
	$(APLT) -s_tail -p_tail $(PRELUDE) network.apl

%.tl: %.apl Makefile
	$(APLT) -p_types -s_tail -c -o $@ $(PRELUDE) $<

%_gen.fut: %.tl Makefile
	$(TAIL2FUT) -o $@ $<
