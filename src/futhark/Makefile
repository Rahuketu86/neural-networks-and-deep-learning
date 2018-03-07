PYTHON=$(shell which python3)
FUTHARK_OPENCL=futhark-opencl
FUTHARK_C=futhark-c

.PHONY: all
all: network.res
	cat $<

.PHONY: small
small: network_small.res
	cat $<

.PHONY: opencl
opencl: network-opencl.res
	cat $<

network-opencl.res: mnist_all.fut.txt network-opencl.exe
	echo 'Image digit prediction percentage:' > $@
	time cat $< | ./network-opencl.exe -t network-opencl.time -d AMD  >> $@
	echo 'Execution time (ex. data load):' >> $@
	cat network-opencl.time >> $@

network.res: mnist_all.fut.txt network.exe
	echo 'Image digit prediction percentage:' > $@
	time cat $< | ./network.exe -t network.time >> $@
	echo 'Execution time (ex data load):' >> $@
	cat network.time >> $@

network_small.res: mnist_small_all.fut.txt network.exe
	echo 'Image digit prediction percentage:' > $@
	time cat $< | ./network.exe -t network_small.time >> $@
	echo 'Execution time (ex data load):' >> $@
	cat network_small.time >> $@

network.exe: network.fut
	$(FUTHARK_C) -o $@ $<

network-opencl.exe: network.fut
	$(FUTHARK_OPENCL) -o $@ $<

.PHONY: prepare
prepare: mnist_all.fut.txt mnist_small_all.fut.txt mnist_training_small_input.apl.txt mnist_training_small_results.apl.txt mnist_test_small_input.apl.txt mnist_test_small_results.apl.txt

mnist_training_input.fut.txt mnist_training_results.fut.txt mnist_test_input.fut.txt mnist_test_results.fut.txt mnist_training_small_input.fut.txt mnist_training_small_results.fut.txt mnist_test_small_input.fut.txt mnist_test_small_results.fut.txt:
	$(PYTHON) mnist_to_fut_format.py

mnist_all.fut.txt: mnist_training_input.fut.txt mnist_training_results.fut.txt mnist_test_input.fut.txt mnist_test_results.fut.txt
	cat $^ > $@

mnist_small_all.fut.txt: mnist_training_small_input.fut.txt mnist_training_small_results.fut.txt mnist_test_small_input.fut.txt mnist_test_small_results.fut.txt
	cat $^ > $@

%.apl.txt: %.fut.txt.tmp
	sed 's/\]//g' $< > $@

%.fut.txt.tmp: %.fut.txt
	sed 's/\[//g' $< > $@

clean:
	rm -rf *~ *.exe *.res *.time

clean_all:
	rm -rf *~ *.fut.txt *.exe *.res
