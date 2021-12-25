include ../make.SciPy.inc

TEST = test_krylov.$(SUFFIX)

default:
	@$(PYTEST) $(TEST)

clean:
	-@$(RM) -r __pycache__

.PHONY: default clean
