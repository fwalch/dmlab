CMAKE_DIR := build
BIN_DIR   := bin

default: release

release: $(CMAKE_DIR) $(BIN_DIR)
	cd $(CMAKE_DIR) && make --no-print-directory

$(CMAKE_DIR):
	mkdir -p $(CMAKE_DIR) && cd $(CMAKE_DIR) && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

purge: clean
	rm -rf $(CMAKE_DIR)

demo: release
	cd bin && ./camera-demo

help:
	@echo "Usage: make <target>"
	@echo "Available targets:"
	@echo "  release        to build in release mode"
	@echo "  clean          to clean build output"
	@echo "  purge          to clean and remove cmake configuration"
	@echo "  help           to show this message"

