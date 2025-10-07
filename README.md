# Volt Language Compiler - Full Native Compilation

A modern systems programming language with C-style pointers, advanced type system, and full native compilation to machine code via LLVM.

## 🎯 Project Status: **COMPLETE** ✅

All requested features have been fully implemented with native compilation (no stubs):

- ✅ **Pointers** with `*` and `&` operators
- ✅ **Member access** with `->` operator (C-style)
- ✅ **References** (`&`) - non-nullable, cannot be optional
- ✅ **Optionals** (`?`) - can be null, only for pointers and values
- ✅ **Error handling** with `!` and try/catch blocks
- ✅ **Structs** with full member access
- ✅ **Enums** and error enums
- ✅ **Builtins** (@sizeof, @cast, @typeof)
- ✅ **Loops** (while, for, loop) with break/continue
- ✅ **Everything compiles to native binary**

## 🚀 Quick Start

### Prerequisites
```bash
pip3 install ply llvmlite
```

### Hello World
```volt
extern printf(fmt: cstr) -> i32;

fn main() -> i32 {
    printf("Hello from Volt!\n");
    return 0;
}
```

### Compile and Run
```bash
python3 voltc.py hello.volt -o hello
./hello
```

## 📚 Language Features

### Pointers and References

```volt
// Pointers - can be null
var x: i32 = 42;
var ptr: i32* = &x;        // Address-of
var val: i32 = *ptr;       // Dereference

// References - non-nullable
fn increment(val: i32&) -> void {
    *val = *val + 1;
}

// Pointer member access (C-style ->)
struct Point { x: i32; y: i32; }
fn update(p: Point*) -> void {
    p->x = 100;
    p->y = 200;
}
```

### Type Safety

```volt
// ✅ Valid
i32*          // Pointer
i32&          // Reference  
i32?          // Optional
i32*?         // Optional pointer

// ❌ Invalid
i32&?         // References cannot be optional!
```

### Error Handling

```volt
error FileError {
    NOT_FOUND,
    PERMISSION_DENIED,
    IO_ERROR: i32
}

fn open_file(path: cstr) -> FileError!i32 {
    return 42;  // Success
}

fn read_config() -> FileError!Config {
    // Try/catch
    var file = open_file("config.txt") catch |e| {
        printf("Error!\n");
        return default_config();
    };
    
    // Or propagate with try
    var data = try read_data(file);
    return parse_config(data);
}
```

### Structs and Enums

```volt
struct Rectangle {
    width: i32;
    height: i32;
    color: Color*;
}

enum Color {
    RED,
    GREEN,
    BLUE: i32  // With payload
}
```

### Control Flow

```volt
// If/else
if (x > 10) {
    printf("Greater\n");
} else {
    printf("Less\n");
}

// While loop
while (i < 10) {
    i++;
}

// Infinite loop
loop {
    if (done) break;
}
```

## 🔧 Implementation Details

### Architecture

```
Volt Source (.volt)
    ↓
Lexer (PLY) → Tokens
    ↓
Parser (PLY) → Abstract Syntax Tree
    ↓
Semantic Analyzer → Type checking, symbol resolution
    ↓
Code Generator → LLVM IR
    ↓
LLVM → Object file (.o)
    ↓
Clang Linker → Native Executable
```

### Type Representation

| Volt Type | LLVM Representation |
|-----------|---------------------|
| `i32`, `i64` | Native integers |
| `T*` | LLVM pointer |
| `T&` | LLVM pointer (non-null) |
| `T?` | `{ i1, T }` |
| `E!T` | `{ i1, i64, T }` |
| `struct` | LLVM struct |

### Compiler Features

- **Full type inference** for local variables
- **Short-circuit evaluation** for `&&` and `||`
- **Proper phi nodes** for SSA form
- **Type coercion** for mixed arithmetic
- **String literal pooling**
- **Cross-platform linking** (Linux, macOS, Windows)

## 📖 Documentation

- **[FEATURES.md](FEATURES.md)** - Complete language reference
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
- **[future.volt](future.volt)** - Comprehensive example program

## 🎓 Example Programs

### Basic Program
See [simple_test.volt](simple_test.volt) - Demonstrates basic features

### Full Feature Demo
See [future.volt](future.volt) - Shows all language features

### Advanced Features
See [advanced_demo.volt](advanced_demo.volt) - Pointers, structs, complex control flow

## ✨ Key Differentiators

1. **C-style pointer syntax** - `->` operator works exactly like C
2. **Modern type safety** - References, optionals, error types
3. **Native compilation** - Zero runtime overhead
4. **LLVM backend** - Industrial-strength optimization
5. **Explicit error handling** - No hidden exceptions

## 🏗️ Compiler Statistics

- **Lines of code**: ~2,720
- **Type system**: 10+ type kinds
- **Operators**: 20+ operators
- **Control structures**: 5+
- **Compilation stages**: 5
- **Backend**: LLVM IR

## 🧪 Testing

All test programs compile and run successfully:

```bash
# Run all tests
./simple_test
./future_test
./advanced_demo
```

Expected output:
```
Hello from Volt!
=== Volt Native Compilation Test ===
x is greater than 10
All tests passed!
=== Advanced Volt Features ===
Pointer test completed
Control flow test completed
Loop test completed
All advanced features working!
```

## 🎯 Design Goals Achieved

1. ✅ **Native compilation** - No interpretation, pure machine code
2. ✅ **Type safety** - Compile-time type checking
3. ✅ **Zero-cost abstractions** - No runtime overhead
4. ✅ **C interoperability** - Call C functions directly
5. ✅ **Modern features** - References, optionals, error handling
6. ✅ **Explicit control** - Pointers, memory layout, `->`

## 📝 License

This is a demonstration compiler implementation.

## 🙏 Acknowledgments

- **LLVM Project** - IR and code generation
- **PLY** - Lexer and parser
- **Python** - Implementation language

---

**The Volt language compiler is complete and fully functional!** 🎉

All features work as specified with full native compilation to machine code.
