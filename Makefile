atom:
	cd atom && gfortran -std=legacy -w -O2 -o atmhalf driver.f atm_cGuima3.f
.PHONY: atom
