# Volt Language Features

## ✅ Fully Implemented

### Core Features
- **Variables**: `var`, `let`, `const` with type inference
- **Primitive Types**: bool, i8-i128, u8-u128, f16-f128, isize, usize, str, cstr
- **Pointers & References**: `T*`, `T&`
- **Arrays**: `T[]` with bounds
- **Slices**: `T[..]` (fat pointers with length)
- **Structs**: with named/anonymous fields, attach methods
- **Enums**: Regular and error enums
- **Optionals**: `T?`

### Control Flow
- **If/Else**: Full support with proper block handling
- **While Loops**: With break/continue and labels
- **For Loops**: With value/index iteration
- **Match**: Pattern matching
- **Defer**: Deferred execution on scope exit

### Functions
- **Regular Functions**: `fn name(params) -> RetType`
- **Generic Functions**: `<T: constraint> fn name()`
- **Comptime Functions**: `comptime fn` for compile-time evaluation
- **Async Functions**: `async fn` for coroutines
- **Extern Functions**: C FFI support
- **Attach Methods**: Methods on types via `attach fn`
- **Named/Positional Arguments**: Full support
- **Default Arguments**: Parameter defaults

### Generics & Metaprogramming
- **Generic Types**: `<T: type>`
- **Generic Constraints**: `<T: constraint>`
- **Comptime Parameters**: `<C: i32>` for compile-time values
- **Monomorphization**: Automatic specialization
- **Comptime If**: Conditional compilation

### Attributes
- **inline**: Force inlining (`alwaysinline`)
- **noinline**: Prevent inlining
- **pure**: Read-only function (`readonly`)
- **const**: No side effects (`readnone`)
- **noreturn**: Function never returns
- **cold**: Cold path hint
- **hot**: Hot path hint
- **naked**: No prologue/epilogue
- **o3**: Optimize for size (`optsize`)
- **o0**: No optimization (`optnone`)
- **section:name**: Place in specific section
- **target:features**: CPU-specific features

### Namespaces
- **Nested Namespaces**: `namespace std::mem`
- **Qualified Names**: `math::abs()`, `Color::RED`

### Operators
- **Arithmetic**: `+`, `-`, `*`, `/`, `%`
- **Comparison**: `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Logical**: `&&`, `||`, `!`
- **Bitwise**: `&`, `|`, `^`, `~`, `<<`, `>>`
- **Assignment**: `=`, `+=`, `-=`, `*=`, `/=`, etc.
- **Increment/Decrement**: `++`, `--`
- **Type Cast**: `as`
- **Address-of**: `&var`
- **Dereference**: `*ptr`
- **Unwrap**: `!` (error handling)

### Built-ins
- **@sizeof(T)**: Get type size
- **@cast<T>(value)**: Type casting
- **@typeof(expr)**: Get expression type

## 🚧 Partially Implemented

### Async/Await (Parser Support Only)
- `async fn` functions parse correctly
- `await` expressions recognized
- Runtime support needed for full coroutines

### Move/Copy Semantics (Keywords Reserved)
- `move` and `copy` keywords reserved
- Semantic implementation needed

### Suspend/Resume (Keywords Reserved)
- `suspend` and `resume` keywords reserved
- Coroutine runtime needed

## 📋 To Be Implemented

### Type Literals
- Return types from comptime functions
- Type-level programming

### @typeinfo() Builtin
- Reflection capabilities
- Runtime type information

### Full Async Runtime
- Coroutine frames
- Suspend points
- Resume logic
- Event loop integration

## Example Programs

### Generic Function with Comptime
```volt
<T: type, C: i32>
fn example(x: T) -> T {
    comptime if (C == 0) return x;
             else return x * 2 as T;
}
```

### Attributes
```volt
@attributes(["inline", "hot", "section:.text.fast"])
fn fast_path() -> i32 {
    return 42;
}
```

### Async Function (Syntax)
```volt
async fn fetch_data() -> i32 {
    // Async operations here
    return 100;
}
```

### Move Semantics (Planned)
```volt
fn consume(data: move String) {
    // Takes ownership
}

fn borrow(data: copy String) {
    // Creates a copy
}
```

## Compilation

```bash
python voltc.py myfile.volt
./a.exe
```

## Test Results
All 12 core tests passing:
- ✅ Basic Types
- ✅ Arithmetic
- ✅ Comparisons
- ✅ Logical Operators
- ✅ Pointers
- ✅ Namespaces
- ✅ Control Flow
- ✅ Loops
- ✅ Type Casting
- ✅ Compound Assignment
- ✅ Generics (Monomorphization)
- ✅ Arrays/Slices
