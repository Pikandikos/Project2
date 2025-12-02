# Minimal Makefile: compile Python files from src/ into build/

PYTHON    ?= python3
SRC_DIR   := src
BUILD_DIR := build

SOURCES := $(wildcard $(SRC_DIR)/*.py)
PYCS    := $(patsubst $(SRC_DIR)/%.py,$(BUILD_DIR)/%.pyc,$(SOURCES))

.PHONY: all compile clean

all: compile

compile: $(PYCS)
	@echo "All Python files compiled into $(BUILD_DIR)/"

# Compile rule: convert src/file.py → build/file.pyc
$(BUILD_DIR)/%.pyc: $(SRC_DIR)/%.py | $(BUILD_DIR)
	@echo "Compiling $< → $@"
	$(PYTHON) -m py_compile $<
	@# Move auto-generated pyc file from __pycache__ to build/
	@pyc_file=$$(basename $< .py); \
	mv $(SRC_DIR)/__pycache__/$$pyc_file.*.pyc $@

# Create build folder if not present
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

clean:
	@echo "Removing build directory..."
	rm -rf $(BUILD_DIR) $(SRC_DIR)/__pycache__
