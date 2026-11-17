PY ?= python3
CORES ?= 16

.PHONY: doctor smoke preflight configs align simulate train predict figures docker

doctor:
	$(PY) workflow/run.py doctor

smoke:
	$(PY) tests/test_smoke.py -v

preflight:
	$(PY) analysis/preflight.py

configs:
	$(PY) workflow/run.py configs --kind all

align:
	$(PY) workflow/run.py align --cores $(CORES)

simulate:
	$(PY) workflow/run.py simulate --role train --cores $(CORES)
	$(PY) workflow/run.py simulate --role test --cores $(CORES)
	$(PY) workflow/run.py simulate --role benchmark --cores $(CORES)

train:
	Rscript workflow/somaticxgb/train_SomaticXGB.R \
	  --pairs $(CPSNV_WORK_ROOT)/4.train/{12,13,21,23,31,32} \
	  --outdir $(CPSNV_WORK_ROOT)/5.somaticxgb/train

predict:
	$(PY) workflow/run.py predict --pairs 45 54

figures:
	$(PY) analysis/07_figures_tables/make_all.py --step all

docker:
	docker build -t cross-plat-somatic-snv .
