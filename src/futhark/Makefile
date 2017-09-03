PYTHON=$(shell which python)
FUTHARK=futhark-opencl
#FUTHARK=futhark-c

.PHONY: all
all: network.res
	cat network.res

network.res: mnist_all.fut.txt network.exe
	echo 'Image digit prediction percentage:' > network.res
	cat $< | ./network.exe -d AMD  >> network.res
#	cat $< | ./network.exe >> network.res

network.exe: network.fut
	$(FUTHARK) -o network.exe network.fut

.PHONY: prepare
prepare: mnist_all.fut.txt

mnist_training_input.fut.txt mnist_training_results.fut.txt mnist_test_input.fut.txt mnist_test_results.fut.txt:
	$(PYTHON) mnist_to_fut_format.py

mnist_all.fut.txt: mnist_training_input.fut.txt mnist_training_results.fut.txt mnist_test_input.fut.txt mnist_test_results.fut.txt
	cat mnist_training_input.fut.txt mnist_training_results.fut.txt mnist_test_input.fut.txt mnist_test_results.fut.txt > mnist_all.fut.txt

clean:
	rm -rf *~ *.exe *.res

clean_all:
	rm -rf *~ *.fut.txt *.exe *.res
