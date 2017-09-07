PYTHON=$(shell which python)
FUTHARK_OPENCL=futhark-opencl
FUTHARK_C=futhark-c

.PHONY: all
all: network.res
	cat $<

.PHONY: opencl
opencl: network-opencl.res
	cat $<

network-opencl.res: mnist_all.fut.txt network-opencl.exe
	echo 'Image digit prediction percentage:' > $@
	time(cat $< | ./network-opencl.exe -t network-opencl.time -d AMD  >> $@)
	echo 'Execution time (ex. data load):' >> $@
	cat network-opencl.time >> $@

network.res: mnist_all.fut.txt network.exe
	echo 'Image digit prediction percentage:' > $@
	time(cat $< | ./network.exe -t network.time >> $@)
	echo 'Execution time (ex data load):' >> $@
	cat network.time >> $@

network.exe: network.fut
	$(FUTHARK_C) -o $@ $<

network-opencl.exe: network.fut
	$(FUTHARK_OPENCL) -o $@ $<

.PHONY: prepare
prepare: mnist_all.fut.txt

mnist_training_input.fut.txt mnist_training_results.fut.txt mnist_test_input.fut.txt mnist_test_results.fut.txt:
	$(PYTHON) mnist_to_fut_format.py

mnist_all.fut.txt: mnist_training_input.fut.txt mnist_training_results.fut.txt mnist_test_input.fut.txt mnist_test_results.fut.txt
	cat mnist_training_input.fut.txt mnist_training_results.fut.txt mnist_test_input.fut.txt mnist_test_results.fut.txt > mnist_all.fut.txt

clean:
	rm -rf *~ *.exe *.res *.time

clean_all:
	rm -rf *~ *.fut.txt *.exe *.res