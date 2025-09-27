// ─────────────────────────────────────────────────────────────────────────────
// Imports
// ─────────────────────────────────────────────────────────────────────────────

use std::foo; // requires linking against the stdlib (on by default but easy to disable in the bolt.toml)
use std::bar; // does not need to be linked against (only makes aware of the namespace, doesnt use it)


// to use a namespace, you need to use the use keyword
// use std::foo::*; // makes it so instead of doing foo::FuncDef, you can just do FuncDef

// var for mutable variables
// let for immutable variables
// const for constants
// fn for functions
// extern for extern functions
// namespace for namespaces
// use for imports
// class for classes
// struct for structs
// enum for enums
// interface for interfaces
// match for pattern matching
// if for conditionals (with else if and else)
// match for match statements (with arms and default)
// for for loops
// while for while loops
// default for default values
// loop for infinite loops
// break for breaking out of loops
// continue for continuing to the next iteration of a loop
// error for errors
// return for returning from a function
// yield for yielding from a function
// await for awaiting from a function
// async for asynchronous functions
// thread for threading
// defer for deferred execution
// suspend for suspending a function
// resume for resuming a function
// static for static variables and functions
// dyn for dynamic variables
// comptime for comptime variables and functions
// volatile for volatile variables
// move for moving variables
// attribute for attributes
// copy for copying variables
// type for type aliases or for defining types in templates
// ref for referencing variables

// CHANGED: casting & reflection
//   - Use reflect<SomeType>() instead of reflect(SomeType)
//   - Use cast<ToType>(expr) instead of `as`

// ADDED: expression blocks
// - if/match/loop/while/block are expressions; last expression (or `break value`) yields a value.

// reflect for reflecting on types and values (only at compile time)
// reflect returns a struct with the following fields, invoked as reflect<some_type>()
/*
    name: str
    type: @type
    type_name: str
    is_unsigned: bool
    methods: @funcdef[]?
    fields: @field[]?
    type_path: @typepath?
    parent_type: @type?
    captures: @field[]? // for closures/lambdas that capture environment
*/

// is for type checking
// in for checking if a value is in a range
// has for checking if a type has a method or field
// and for logical and
// or for logical or
// bor for bitwise or
// band for bitwise and
// bxor for bitwise xor
// bnot for bitwise not
// bshl for bitwise shift left
// bshr for bitwise shift right

// ADDED: pointer member/method access via `->`
// - p->field   == (*p).field
// - p->method(...) == (*p).method(...)

// AST types: (more to come in the future)
// ast::FuncDef, ast::Type, ast::Param, 
// ast::Block, ast::Stmt,
// ast::ExprStmt, ast::TryBlock, 
// ast::CatchBlock, ast::Call, 
// ast::Expr, ast::Num, ast::Bool, 
// ast::Str, ast::VarDecl, ast::AssignStmt, ast::ReturnStmt
// these are only accessible within the macro context AND only at compile time
// ADDED (compile-time only surface):
// ast::Lambda, ast::Match, ast::IfExpr, ast::LoopExpr, ast::CatchExpr
// ast::FieldAccess, ast::PtrFieldAccess, ast::MethodCall, ast::PtrMethodCall

// Builtins:
// Builtins are functions that are built into the language and are not user-defined
// Builtins are always available and are not subject to the visibility rules
// @set_bit(var, bit, value)
// @clear_bit(var, bit)
// @toggle_bit(var, bit)
// @get_bit(var, bit)
// @set_bits(var, bits)
// @clear_bits(var, bits)
// @toggle_bits(var, bits)
// @attribute(attribute)
// @attributes(attribute[])
// builtin attributes include:
// "inline", "noinline", "always_inline", "never_inline",
// "optimize", "optimize2", "optimize3", "optimize_size", "nooptimize",
// "naked", "section", "address"
// builtin types:
// @type
// @funcdef
// @param
// @block
// @stmt
// @exprstmt
// @tryblock
// @catchblock
// 

// types:
// i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, f16, f32, f64, f128, bool, str, charptr, void, usize, isize
// type, type*, type& 

// ADDED: tuples
// - unnamed tuples: (i32, str) — access via t[0], t[1]
// - named tuples:   (x: i32, y: i32) — access via t.x, t.y  (indexing t[0], t[1] also allowed)

// ADDED: closures/lambdas
// - |x: T, y: U| -> R { body }   (captures inferred; optional capture list [move a, &b])

// ADDED: try/catch expressions (zig-like)
// - let v = try foo();                // propagate error
// - let v = foo() catch 0;            // default on error
// - let v = foo() catch (e) { ... };  // handler produces a value



// ─────────────────────────────────────────────────────────────────────────────
// Macros (yours, unchanged)
// ─────────────────────────────────────────────────────────────────────────────

public macro change_return_type(
    func: @funcdef,
    new_type: @type
) {
    func.ret_type = new_type;
    return func;
}

public macro add_param(
    func: @funcdef,
    param_name: str,
    param_type: @type
) {
    func.params.append(@param(name=param_name, type=param_type));
    return func;
}

public macro rename_func(
    func: @funcdef,
    new_name: str
) {
    func.name = new_name;
    return func;
}

public macro inject_at_start(
    func: @funcdef,
    stmt: @stmt
) {
    func.body.stmts.insert(0, stmt);
    return func;
}

public macro wrap_with_try(
    func: @funcdef
) {
    let old_body = func.body;
    func.body = @block([
        @tryblock(
            try_body=old_body,
            catch_body=@block([
                @exprstmt(@call("handle_error", []))
            ])
        )
    ]);
    return func;
}

public macro set_attribute(
    type_or_value: @type,
    attribute: @attribute
) {
    type_or_value.attributes.append(attribute);
    return type_or_value;
}

// Example usage (pseudo-syntax):
// let f = change_return_type!(my_func, @type("i64"));
// let f2 = add_param!(f, "extra", @type("i32"));


// ─────────────────────────────────────────────────────────────────────────────
// Visibility constants 
// ─────────────────────────────────────────────────────────────────────────────

const EXAMPLE_CONST: i32 = 1;
const EXAMPLE_CONST_2: i32 = 2;
const EXAMPLE_CONST_3: i32 = 3;
const EXAMPLE_CONST_4: i32 = 4;
const EXAMPLE_CONST_5: i32 = 5;
const EXAMPLE_CONST_6: i32 = 6;
const EXAMPLE_CONST_7: i32 = 7;


// ─────────────────────────────────────────────────────────────────────────────
// Errors + zig-like try/catch expressions (yours + added usage)
// ─────────────────────────────────────────────────────────────────────────────

public enum(error) some_error {
    some_error,
    some_other_error,
    yet_another_error,
    one_more_error,
    last_error,
}

const some_condition: bool = false;

public fn some_func_that_might_error() -> i32!some_error {
    if some_condition { return some_error::some_error; }
    return 1;
}

public fn some_func_that_might_error_2() -> i32! {
    if some_condition { return error; }
    return 1;
}

public fn use_try_catch() -> i32!some_error {
    let a: i32 = try some_func_that_might_error();
    let b: i32 = some_func_that_might_error_2() catch 0;
    let c: i32 = some_func_that_might_error() catch (e) { 123 };
    return a + b + c;
}


// ─────────────────────────────────────────────────────────────────────────────
// Data types  + lifecycle/allocator + pointer `->`
// ─────────────────────────────────────────────────────────────────────────────

public enum some_enum<T: type> {
    some_enum,
    some_other_enum,
    yet_another_enum(i32, f32),
    one_more_enum(i32),
    last_enum(T),
}

public struct some_data {
    a: i32,
    b: i32,
    c: i32,
}

// ADDED: Allocator interface + default allocator (malloc-like)
public interface allocator {
    public fn alloc(size: usize, align: usize) -> usize*;
    public fn free(ptr: usize*) -> void;
}

public class default_allocator : allocator {
    public fn alloc(size: usize, align: usize) -> usize* { return SOME_DEFAULT_ALLOC(size, align); }
    public fn free(ptr: usize*) -> void { SOME_DEFAULT_FREE(ptr); }
}

// classes should not be used in embedded systems as
// they have hidden control flow (constructors, destructors, etc.)
public class some_class {

    private _some_data: some_data;
    public some_data: some_data;

    // CHANGED: require explicit `this` for instance methods; no `this` means static.
    public fn some_method(this: some_class) {
        return this._some_data;
    }
}

// ADDED: All classes define construct/destruct if they manage resources.
// - construct can be called implicitly via `TypeName(...)`
// - if omitted, a trivial default construct exists (value-init).
public class file_buf<A: allocator = default_allocator> {
    private _ptr: usize*;
    private _len: usize;
    private _alloc: A;

    // static construct returning HEAP pointer (implicit call enabled)
    // this will error out if A is not defaulted
    public static fn construct(alloc: A = A{}, len: usize) -> file_buf<A>* {
        var mem = alloc.alloc(len, @alignof<u8>());
        var self = file_buf<A>{ _ptr = mem, _len = len, _alloc = alloc };
        var obj_mem = alloc.alloc(@sizeof<file_buf<A>>(), @alignof<file_buf<A>>());
        *(file_buf<A>*)obj_mem = self;
        return (file_buf<A>*)obj_mem;
    }

    // overloaded constructor will error out if the variable using it isnt explicit like:
    // var fb = file_buf(128); <-- this will error out
    // var fb: file_buf = file_buf(128); <-- this will work
    public fn construct(alloc: A = A{}, len: usize) -> file_buf<A> {
        var mem = alloc.alloc(len, @alignof<u8>());
        return file_buf<A>{ _ptr = mem, _len = len, _alloc = alloc };
    }

    // instance destructor (must be called before freeing object storage)
    public fn destruct(this: file_buf<A>*) -> void {
        this->_alloc.free(this->_ptr);
        this->_alloc.free((usize*)this);
    }

    public fn len(this: file_buf<A>*) -> usize { return this->_len; }
}

// ADDED: Implicit construct call sugar
//   var p = file_buf<default_allocator>(default_allocator{}, 1024);
// expands to:
//   var p = file_buf<default_allocator>::construct(default_allocator{}, 1024);
//
// NOTE: Implicit construct for classes that return pointers yields a *pointer*.
// Access fields/methods through `->`.


// using custom allocator (structural constraint is OK)
public class some_class<A: type> 
where:
    A has alloc(size: i32) -> usize*
{
    private _allocator: A;
    public fn some_method(this: some_class<A>, allocator: A) {
        // example only
        this._some_data = cast<some_data*>(allocator.alloc(@sizeof<some_data>(), @alignof<some_data>()));
    }
}

// interfaces
public interface some_interface {
    public some_data: some_data;
    public fn some_method(this: some_interface) -> i32; // requires `this`
}

// implementations
public class some_class2 : some_interface {
    public some_data: some_data;
    public fn some_method(this: some_class2) -> i32 {
        return this.some_data.a + this.some_data.b + this.some_data.c;
    }
}

// generics
public fn some_generic_func<T: type>(a: T) -> T { return a; }

public fn some_generic_func_2<T: type, B: i32>(a: T) -> T! 
where: 
    B > 0,
    T is some_interface,
    T has some_method
{ return a; }


// ─────────────────────────────────────────────────────────────────────────────
// Ownership, destruct, and `defer`
// ─────────────────────────────────────────────────────────────────────────────
//
// Safety tiers:
//  - `safe` (default) enforces borrow rules and prevents aliasing mutable references.
//  - `unsafe` blocks allow raw pointer mutation and manual lifetime management.
// 1) Stack values: destructor runs automatically at scope exit (if type has one).
// 2) Heap objects (Class* returned by construct/implicit call):
//    - You must ensure `destruct()` is called exactly once before the object storage is freed.
//    - Shorthand: `defer obj;` schedules the right thing:
//         - if `obj` is a value with destructor: call value’s destructor at scope exit.
//         - if `obj` is a Class* with `destruct`: expand to `defer obj->destruct();`
//    - `borrow obj` creates a read-only borrow; `borrow mut obj` grants unique mutable access.
// 3) Moves: after `move x`, callee owns `x` and is responsible for destroying it (if needed).

public fn defer_demo() {
    // HEAP-allocated class via implicit construct
    var fb = file_buf<default_allocator>(default_allocator{}, 256);
    defer fb;                   // schedules fb->destruct() at scope exit
    let n = fb->len();

    borrow mut fb_mut = fb;     // mutable borrow within this scope
    fb_mut->_len = n + 1;

    // STACK tuple with named fields
    let t: (x: i32, y: i32) = (x: 2, y: 3);
    let s = t.x + t.y;

    // UNNAMED tuple
    let u: (i32, i32) = (4, 5);
    let s2 = u[0] + u[1];

    // if / match / loop as expressions
    let v = if (n > 0) { n } else { 0 };

    let m = match n {
        0 => 0,
        k in 1..=10 => k * 2,
        default => -1,
    };

    let acc = loop {
        if (m > 20) { break m; }
        break m + 1;
    };
}

// Moving ownership example
public fn takes_ownership(buf: file_buf<default_allocator>*) {
    defer buf; // callee now responsible
    // use buf...
}

public fn unsafe_pointer_edit(buf: file_buf<default_allocator>*) {
    unsafe {
        buf->_len = 0; // allowed because we're in unsafe block
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// comptime 
// ─────────────────────────────────────────────────────────────────────────────

// comptime name: type = value;

// public comptime fn name(params) -> ret_type {
//     body
// }

public comptime fn double_bytes(len: usize) -> usize {
    return len * 2;
}

public fn comptime_demo() -> usize {
    comptime base: usize = 32;
    comptime doubled: usize = double_bytes(base);
    return doubled;
}


// ─────────────────────────────────────────────────────────────────────────────
// Async/threads 
// ─────────────────────────────────────────────────────────────────────────────

// public async fn some_async_fn() { await some_async_fn_2(); }
// public thread fn some_thread_fn() { let handle = thread some_async_fn(); await handle; }

// Runtime model:
//  - async functions suspend on await and run on the process-wide executor (configurable via bolt.toml).
//  - `thread` creates an owned OS thread; caller must `await`/`join` the handle or explicitly detach.
//  - Cancellation propagates top-down: cancelling a parent task signals children before unwinding.
//  - Structured concurrency: use `async scope { ... }` to ensure spawned tasks finish before scope exit.

public async fn async_square(value: i32) -> i32 {
    return value * value;
}

public async fn cancellable_delay(ms: usize, token: cancel_token) -> void! {
    while (!token.is_cancelled()) {
        await runtime::sleep(ms);
        break;
    }
    return;
}

public fn spawn_async_task() -> i32 {
    let handle = thread async_square(12);
    let result = await handle;
    return result;
}

public async fn structured_spawn_demo() -> i32! {
    let scope_result = async scope (token) {
        let worker = async spawn { return await async_square(5); };
        let watcher = async spawn { await cancellable_delay(10, token); };
        let value = await worker;
        // watcher is automatically cancelled when scope exits if still running
        await watcher catch { 0 };
        return value;
    };
    return scope_result;
}

// suspend and resume 
// public fn some_fn() {
//     { body
//       suspend;
//       body <-- execution resumes here after suspend
//       some_label: <-- or here explicitly if called }
// }
// public fn some_fn_2() { 
//    some_fn();
//    resume some_label: some_fn(); 
// }

public fn suspendable_block() {
    {
        suspend;
        resume_point:
        return;
    }
}

public fn drive_suspend() {
    suspendable_block();
    resume resume_point: suspendable_block();
}


// ─────────────────────────────────────────────────────────────────────────────
// Loops  + note on expressions
// ─────────────────────────────────────────────────────────────────────────────

// for / while / loop as before; remember they can yield values via `break value`.

// loops can be labeled as before.


// ─────────────────────────────────────────────────────────────────────────────
// Attributes 
// ─────────────────────────────────────────────────────────────────────────────

// attribute name = some_value; // any custom attribute macro or comptime value returning @attribute

// Attribute evaluation rules:
//  - Declarative macros must return @attribute values (eg. cold_section!()).
//  - Raw comptime values that are already @attribute can be listed directly.
//  - Attributes are validated at compile time; duplicates merge by last-wins precedence.
//  - Builtin shorthand: use identifier directly for no-arg attributes (inline, optimize3, naked, ...)
//    and call like a function when arguments are required (section("startup"), address(0x08000000)).
//  - Order matters: later attributes override conflicting earlier ones on the same key.

@attributes([inline, optimize3])
public fn attributed_example(x: i32, y: i32) -> i32 {
    return x + y;
}

@attribute(naked)
public fn attributed_naked_handler() {
    asm "generic" {
        "// naked handler body";
    };
}

@attributes([section("startup"), address(0x08000000)])
public fn startup_entrypoint() {
    // This function will be emitted in the "startup" section at 0x08000000.
}

// Custom attributes can come from macros that return @attribute.
public macro cold_section() {
    return attribute(name: "section", args: "cold");
}

@attributes([inline, cold_section!()])
public fn macro_attribute_demo() {
    // Demonstrates macro-produced custom attribute usage.
}

@attributes([optimize_size, optimize3])
public fn attribute_precedence_demo() -> i32 {
    // optimize3 overrides optimize_size due to ordering (last wins).
    return 42;
}

// @attributes([unknown_attribute]) <-- compile-time error: validator rejects undefined attribute.

// ─────────────────────────────────────────────────────────────────────────────
// Inline assembly
// ─────────────────────────────────────────────────────────────────────────────

// Syntax:
// asm "target", [options] {
//     "assembly text";
// } (outputs, inputs, clobbers?);
// Operand classes: `in`, `out`, `inout`, `const`, `sym` for symbol operands.
// Clobbers accept registers or "memory"; diagnostics warn on unsupported names.
// Target resolution errors if architecture string unknown or instruction not supported.
// Prefer high-level intrinsics in std::intrinsics before dropping to asm.

public fn asm_add(lhs: i32, rhs: i32) -> i32 {
    var result: i32 = 0;
    asm "generic" {
        "add {out}, {in1}, {in2}";
    } (out result, in lhs, in rhs);
    return result;
}

public fn asm_memory_barrier() {
    asm "generic", [volatile] {
        "mfence";
    } (clobber "memory");
}

// target-specific example placing inline asm inside naked function.
@attribute(attribute(name: "naked", args: null))
public fn naked_asm_stub() {
    asm "armv7", [volatile] {
        "push {lr}";
        "bl some_runtime_handler";
        "pop {pc}";
    } (clobber "memory");
}

public fn asm_constrained_example(x: i32) -> i32 {
    var out_val: i32 = x;
    asm "generic" {
        "mul {inout0}, {inout0}, 2";
    } (inout out_val, clobber "r0");
    return out_val;
}

public fn prefer_intrinsic_example(x: i32) -> i32 {
    // std::intrinsics::rotate_left maps to efficient instructions when available.
    return std::intrinsics::rotate_left(x, 3);
}


// ─────────────────────────────────────────────────────────────────────────────
// reflect & cast (CHANGED forms) + example
// ─────────────────────────────────────────────────────────────────────────────

public fn reflect_and_cast_demo() {
    let info = reflect<i64>();                 // CHANGED: reflect<type>()
    let aligned = @alignof<i64>();
    let sz = @sizeof<i64>();
    let p: usize* = @default_alloc(64, @alignof<u8>());
    let pi: i32* = cast<i32*>(p);              // CHANGED: cast<T>(expr)

    @default_free(p);
    // tuples:
    let pair: (left: i32, right: i32) = (left: 1, right: 2);
    let sum = pair.left + pair.right;
}


// ─────────────────────────────────────────────────────────────────────────────
// extern/export/namespaces 
// ─────────────────────────────────────────────────────────────────────────────

// extern "ABI" fn name(params) -> ret_type;
// export "ABI" fn name(params) -> ret_type;

// namespace name {
//     items
// }

// package metadata:
// package demo {
//     version = "1.0.0";
//     authors = ["user@example.com"];
// }

// visibility modifiers:
// public/export/internal/private available on declarations.

// variable declarations:

// const name: type = value;
// let name: type = value;
// static name: type = value;
// comptime name: type = value;
// volatile name: type = value;
// attribute name = value;
// dyn some_dynamic_type = value;

// export use namespaces:
// export use namespace::*; // (be cautious)

// use namespaces:
// use namespace::*; // imports all items recursively (tree-shaken)
// use namespace::{Item1, Item2}; // selective imports
// use namespace::submodule::{self, Item3 as Alias}; // granular imports

extern fn c_puts(message: str) -> void;

namespace math_util {
    public fn add(a: i32, b: i32) -> i32 {
        return a + b;
    }
}

use math_util::add;

namespace math_util::advanced {
    public fn mul(a: i32, b: i32) -> i32 { return a * b; }
}

use math_util::advanced::{mul as multiply, self};

export "c" fn exported_sum(a: i32, b: i32) -> i32 {
    let result = add(a, b);
    let doubled = multiply(result, 2);
    c_puts("exported_sum called");
    return result;
}


// ─────────────────────────────────────────────────────────────────────────────
// Function pointers + closures (yours + allowed)
// ─────────────────────────────────────────────────────────────────────────────

// fn name<GENERICS>(params) -> ret_type;
// fn name<GENERICS>(params) -> ret_type!ERROR;
// closure types are accepted where function pointers appear.

public fn apply_binary(a: i32, b: i32, op: fn(i32, i32) -> i32) -> i32 {
    return op(a, b);
}

public fn function_pointer_demo() -> i32 {
    let multiplier = |x: i32, y: i32| -> i32 { x * y };
    return apply_binary(6, 7, multiplier);
}


// ─────────────────────────────────────────────────────────────────────────────
// Pointers & semantics  + `->`
// ─────────────────────────────────────────────────────────────────────────────

// *pointer;    // deref
// &x;          // address
// p->field;    // sugar for (*p).field
// p->method(); // sugar for (*p).method()

// move/copy/ref rules remain as you wrote.

public fn pointer_usage_demo() -> i32 {
    var local = some_data{ a = 1, b = 2, c = 3 };
    let ptr: some_data* = &local;
    ptr->a = 10;
    let sum = ptr->a + ptr->b + ptr->c;
    return sum;
}


// ─────────────────────────────────────────────────────────────────────────────
// Builtins (formal signatures, not actually callable like this, just for documentation)
// ─────────────────────────────────────────────────────────────────────────────

public struct attribute { name: str, args: any? }

// bitwise (integer only)
extern builtin @set_bit   <T: type>(var: T&, bit: usize, value: bool) -> void where: T is integer;
extern builtin @clear_bit <T: type>(var: T&, bit: usize) -> void where: T is integer;
extern builtin @toggle_bit<T: type>(var: T&, bit: usize) -> void where: T is integer;
extern builtin @get_bit   <T: type>(var: T,  bit: usize) -> bool where: T is integer;

extern builtin @set_bits   <T: type>(var: T&, bits: T) -> void where: T is integer;
extern builtin @clear_bits <T: type>(var: T&, bits: T) -> void where: T is integer;
extern builtin @toggle_bits<T: type>(var: T&, bits: T) -> void where: T is integer;

// attributes
extern builtin @attribute (attr: attribute) -> void;
extern builtin @attributes(attrs: attribute[]) -> void;

// default allocator helpers
extern builtin @default_alloc(bytes: usize, align: usize) -> usize*;
extern builtin @default_free(ptr: usize*) -> void;

// introspection/casts
extern builtin @sizeof<T: type>() -> usize;
extern builtin @alignof<T: type>() -> usize;
// reflect is a compile-time intrinsic used as reflect<T>()
extern builtin reflect<T: type>() -> /* ReflectInfo */ any;
// cast is a builtin generic function
extern builtin cast<T: type>(value: any) -> T!;  // fallible reinterpret/convert

public fn builtin_bit_examples() -> bool {
    var flags: u8 = 0;
    @set_bit(flags, 1, true);
    @toggle_bit(flags, 2);
    let is_set = @get_bit(flags, 1);
    @clear_bit(flags, 2);
    return is_set;
}


// ─────────────────────────────────────────────────────────────────────────────
// Small end-to-end sample tying the changes together
// ─────────────────────────────────────────────────────────────────────────────

public fn everything_demo() -> i32! {
    // implicit construct (heap) + defer
    var fb = file_buf(default_allocator{}, 128); // or file_buf(128) <-- this only works if the type that isnt defaulted is different then the ones being defaulted if they are the first parameter (otherwise error for ambiguous call)
    defer fb;

    // closure capturing by ref
    let mult = |a: i32, b: i32| -> i32 { a * b };
    let info = reflect<mult>();

    // tuples
    let pt: (x: i32, y: i32) = (x: 3, y: 4);
    let sum1 = pt.x + pt.y;

    let dims: (i32, i32) = (5, 6);
    let sum2 = dims[0] + dims[1];

    // try/catch as expressions
    let base = some_func_that_might_error_2() catch 7;

    // control flow as expressions
    let scaled = if (sum1 > 0) { sum1 * base } else { 0 };

    // loop as expression
    let out = loop {
        if (scaled > 20) { break scaled; }
        break scaled + 1;
    };

    return out;
}
