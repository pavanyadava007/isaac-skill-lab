# Isaac Sim 4.5 pip wheels + Isaac Lab 2.1.1 in an existing venv (see README "Setup")
IPY ?= ../humanoid-loco/.venv-isaac/bin/python
ISAAC = OMNI_KIT_ACCEPT_EULA=YES timeout -s KILL 3600 $(IPY)

.PHONY: smoke collect train tune eval diagnose video report space gate lint test all

smoke:
	$(ISAAC) scripts/smoke.py --headless --num_envs 16

collect:
	$(ISAAC) scripts/collect.py --headless --rand dr --num_envs 1024
	$(ISAAC) scripts/collect.py --headless --rand no_dr --num_envs 1024

train:
	$(IPY) scripts/train.py --grid scaling --grid dr_vs_nodr --grid rnn

tune:
	$(ISAAC) scripts/tune_controller.py --headless

eval:
	$(ISAAC) scripts/evaluate.py --headless --world dr
	$(ISAAC) scripts/evaluate.py --headless --world small_cubes
	$(ISAAC) scripts/evaluate.py --headless --world no_dr

diagnose:
	$(ISAAC) scripts/diagnose_failures.py --headless

video:
	$(ISAAC) scripts/render_video.py --headless --enable_cameras --policy mlp_dr_n800_s0

report:
	$(IPY) scripts/make_report.py

space:
	$(IPY) scripts/build_space.py

all: collect train tune eval diagnose video report

lint:
	uvx ruff check .

test:
	$(IPY) -m pytest -q tests

gate: lint test
