CC ?= gcc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -Wpedantic -Wconversion -Werror
BUILD ?= ../build

all: $(BUILD)/libe2e_guard.so $(BUILD)/test_e2e_guard

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/libe2e_guard.so: e2e_guard.c e2e_guard.h | $(BUILD)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ e2e_guard.c

$(BUILD)/test_e2e_guard: test_e2e_guard.c e2e_guard.c e2e_guard.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ test_e2e_guard.c e2e_guard.c

test: $(BUILD)/test_e2e_guard
	$(BUILD)/test_e2e_guard

clean:
	rm -rf $(BUILD)/libe2e_guard.so $(BUILD)/test_e2e_guard

.PHONY: all test clean
