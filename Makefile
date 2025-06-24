# ── toolchain ──────────────────────────────────────────────────────────────
CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iinclude

# ── targets & sources ──────────────────────────────────────────────────────
BIN      := xor_net
SRC      := app/main.cpp

.PHONY: all run clean

# default target
all: $(BIN)

# link/compile step (only one .cpp in this header-only setup)
$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

# convenience: `make run` builds (if needed) then executes
run: $(BIN)
	./$(BIN)

# remove build artifacts
clean:
	$(RM) $(BIN)
