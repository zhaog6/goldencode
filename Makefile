DIRS = test demo

default:
	@for i in $(DIRS); \
		do \
		$(MAKE) -C $$i; \
		done

clean:
	@for i in $(DIRS); \
		do \
		$(MAKE) -C $$i clean; \
		done

.PHONY: default, clean
