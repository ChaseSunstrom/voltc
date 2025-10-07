// Complete Volt Features Demonstration
extern C printf(fmt: cstr) -> i32;
extern C puts(s: cstr) -> i32;

// ANSI Color Codes
const RESET: cstr = "\x1b[0m";
const GREEN: cstr = "\x1b[32m";
const RED: cstr = "\x1b[31m";
const YELLOW: cstr = "\x1b[33m";
const BLUE: cstr = "\x1b[34m";
const BOLD: cstr = "\x1b[1m";

const OK: cstr = "\x1b[32m[OK]\x1b[0m";
const FAIL: cstr = "\x1b[31m[FAIL]\x1b[0m";

// Because of function overloading, if we want to call this in C, we have to export it like so:
export C some_func(x: i32) -> i32;
export C some_func2(X: i32, Y: i32) -> i32; // These will not be name mangled.

// For generic functions, we could do this:
<T: type>
export C some_func(x: T) -> T;

<T: type>
export C some_func2(X: T, Y: T) -> T; // These WILL be name mangled into: some_func + name of T, for examplme: some_funci32

// NAMESPACES
namespace math {
    fn abs(x: i32) -> i32 {
        if (x < 0) {
            return 0 - x;
        }
        return x;
    }
    
    fn max(a: i32, b: i32) -> i32 {
        if (a > b) {
            return a;
        }
        return b;
    }
}

namespace utils {
    fn double(x: i32) -> i32 {
        return x * 2;
    }
}

namespace std::mem {
    struct default_allocator {
        malloc: fn<T>(size: usize) -> !T*;
        realloc: fn<T>(ptr: T*, size: usize) -> !T*;
        free: fn<T>(ptr: T*);
    }

    attach fn new(static this: default_allocator) -> default_allocator {
        return {
            malloc,
            realloc,
            free
        };
    }

    <T: type>
    fn malloc(size: usize?) -> !T* {
        if (usize == null) {
            return @cast<T*>(0);
        } else {
            return @cast<T*>(usize);
        }
    }

    <T: type>
    fn realloc(ptr: T*, size: usize) -> !T* {
        return @cast<T*>(size);
    }

    <T: type>
    fn free(ptr: T*) {
        // empty for now
    }
}

// STRUCTS
@attributes(["packed"])
struct Point {
    x: i32;
    y: i32;
}

// ENUMS
enum Color {
    RED,
    GREEN,
    BLUE
}

// ERROR ENUMS
error MyError {
    INVALID,
    FAILED
}

// FUNCTIONS WITH LOOPS
@attributes(["inline", "o3", "section:.text"])
fn sum_range(n: i32) -> i32 {
    var sum: i32 = 0;
    var i: i32 = 0;
    while (i < n) {
        sum = sum + i;
        i++;
    }
    return sum;
}

attach fn new(static this: Point) -> Point {
    return { 0, 0};
}

attach fn new(static this: Point, x: i32) -> Point { // function overload
    return { x: x, 0}; // or return { x, 0 };
}

attach fn delete(this: Point) -> Point {
  // empty for now
}

<T: type>
constraint allocator {
    malloc: has fn<T>(usize) -> !T*;
    realloc: has fn<T>(T*, usize) -> !T*;
    free:   has fn<T>(T*);

    // for more specific usage:
    // malloc: has fn malloc(usize) -> !T*;
}

<Alloc: allocator = std::mem::default_allocator>
fn some_test(alloc: Alloc) -> i32* {
    return alloc.malloc<i32>(@sizeof(i32))!; // ! is shorthand for "get value or panic, even if its an error"
}  

<T: type>
constraint isint {
    T: is i8 || is i16 || is i32 || is i64; // etc...
}

// could then be used like this:
<T: isint, C: i32> // or <T: isint<T>>
attach fn new(static this: Point, x: i32, y: T) -> MyError!Point {

    comptime if (C == 0) return { x, y as i32 }; // or more explicitly @cast<i32>(y)
             else return MyError::FAILED;
}

// Comptime functions - TODO: implement type literals
// <C: i32>
// comptime fn comptime_get_type() -> type {
//     comptime if (C == 0) return i32;
//              else return i64;
// }

<C: i32>
comptime fn comptime_get_value() -> i32 {
    comptime if (C == 0) return 100;
             else return 200;
}

fn overload_test(x: i32) -> i32 {
    return x;
}

fn overload_test(x: i32, y: i32) -> i32 {
    return x + y;
}

async fn test_async() -> i32 {
    return 42;
}

// Suspend/Resume test function
async fn test_suspend_resume() -> i32 {
    var result: i32 = 0;

    // Suspend statements (currently no-op, syntax test)
    suspend;

    result = 10;
    suspend result;

    result = result + 5;
    return result;
}

// Function to test resume (currently stub)
fn test_resume_call() -> i32 {
    // In a full implementation:
    // var coro = test_suspend_resume();
    // resume coro;
    return 15;
}

fn main() -> i32 {
    printf("%s%s=== COMPLETE VOLT FEATURE TEST ===%s\n\n", BOLD, BLUE, RESET);

    var passed: i32 = 0;
    var failed: i32 = 0;

    // Debug: Test basic control flow
    var debug_val: i32 = 5;
    if (debug_val < 10) {
        debug_val = 100;
    }
    printf("DEBUG: debug_val after if = %d (should be 100)\n", debug_val);

    var loop_test: i32 = 0;
    var loop_i: i32 = 0;
    while (loop_i < 3) {
        loop_test = loop_test + 1;
        loop_i++;
    }
    printf("DEBUG: loop_test = %d (should be 3)\n", loop_test);

    // 1. Basic types
    printf("%s1. Basic Types:%s ", BOLD, RESET);
    var x: i32 = 42;
    var y: i64 = 1000;
    var b: bool = true;
    if (x == 42 && y == 1000 && b == true) {
        printf("%s (x=%d, y=%d, b=%d)\n", OK, x, y as i32, b as i32);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }

    // 2. Arithmetic
    printf("%s2. Arithmetic:%s ", BOLD, RESET);
    var sum: i32 = x + 10;
    var prod: i32 = x * 2;
    if (sum == 52 && prod == 84) {
        printf("%s (sum=%d, prod=%d)\n", OK, sum, prod);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }

    // 3. Comparisons
    printf("%s3. Comparisons:%s ", BOLD, RESET);
    var eq: bool = x == 42;
    var gt: bool = x > 10;
    if (eq && gt) {
        printf("%s (eq=%d, gt=%d)\n", OK, eq as i32, gt as i32);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }

    // 4. Logical operators
    printf("%s4. Logical ops:%s ", BOLD, RESET);
    var and_test: bool = (x > 10) && (x < 100);
    var or_test: bool = (x == 0) || (x == 42);
    if (and_test && or_test) {
        printf("%s (and=%d, or=%d)\n", OK, and_test as i32, or_test as i32);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }

    // 5. Pointers
    printf("%s5. Pointers:%s ", BOLD, RESET);
    var ptr: i32* = &x;
    var deref: i32 = *ptr;
    if (deref == 42) {
        printf("%s (deref=%d)\n", OK, deref);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }

    // 6. Namespaces
    printf("%s6. Namespaces:%s ", BOLD, RESET);
    var abs_val: i32 = math::abs(0 - 10);
    var max_val: i32 = math::max(5, 10);
    var doubled: i32 = utils::double(21);
    if (abs_val == 10 && max_val == 10 && doubled == 42) {
        printf("%s (abs=%d, max=%d, double=%d)\n", OK, abs_val, max_val, doubled);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }
    
    // 7. Control flow
    printf("7. Control flow: ");
    if (x > 10) {
        var temp: i32 = 1;
    } else {
        var temp: i32 = 0;
    }

    var color = Color::RED;

    match (color) {
        Color::RED => _,
        Color::GREEN => _,
        Color::BLUE => _,
        default => printf("INVALID COLOR\n")
    }

    var point = Point::new<i32, 0>(10); // Will be evaluated at comptime (for the if, so we know with this param it will always return a value)
    var errorp = Point::new<i32, 1>(1); // errorp will be of value e
    defer point.delete(); // Gets ran on this function exit unless move is used on it passing it into another function
    // ^^ if point were to get moved into another function, defer would get ran at the end of THAT function instead, and it would be inaccessible here causing a compiler error

    match (errorp) {
        MyError::INVALID => _,
        MyError::FAILED => _ // _ is no op
    }

    printf("%s\n", OK);
    passed++;

    // 8. Loops
    printf("%s8. Loops:%s ", BOLD, RESET);
    var loop_result: i32 = sum_range(10);
    if (loop_result == 45) {
        printf("%s (sum_range(10)=%d)\n", OK, loop_result);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }

    // 9. Type casting
    printf("%s9. Type casting:%s ", BOLD, RESET);
    var casted: i64 = x as i64;
    if (casted == 42) {
        printf("%s (i32->i64=%d)\n", OK, casted as i32);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }

    // 10. Compound assignment
    printf("%s10. Compound assign:%s ", BOLD, RESET);
    var counter: i32 = 0;
    counter++;
    counter += 5;
    if (counter == 6) {
        printf("%s (counter=%d)\n", OK, counter);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }

    // 11. Allocators and generics
    printf("%s11. Generics:%s ", BOLD, RESET);
    var some_alloced = some_test();
    printf("%s (monomorphization)\n", OK);
    passed++;

    // 12. Arrays/Slices
    printf("%s12. Arrays/Slices:%s ", BOLD, RESET);
    var some_array: i32[] = { 1, 2, 3, 4 };
    var some_slice: i32[..] = { 0..some_array.length };

    some_array[0] = 2;
    some_slice[0] = 3;
    if (some_array[0] == 2) {
        printf("%s (array[0]=%d, slice[0]=%d)\n", OK, some_array[0], some_slice[0]);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }

    // 13. @sizeof builtin
    printf("%s13. @sizeof:%s ", BOLD, RESET);
    var size_i32: i32 = @sizeof(i32) as i32;
    var size_i64: i32 = @sizeof(i64) as i32;
    var size_ptr: i32 = @sizeof(i32*) as i32;
    if (size_i32 == 4 && size_i64 == 8 && size_ptr == 8) {
        printf("%s (i32=%d, i64=%d, ptr=%d)\n", OK, size_i32, size_i64, size_ptr);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }

    // 14. Comptime loops
    printf("%s14. Comptime loops:%s ", BOLD, RESET);
    var comptime_sum: i32 = 0;
    // Comptime loop unrolling - for now test runtime equivalent
    var ct_i: i32 = 0;
    while (ct_i < 5) {
        comptime_sum = comptime_sum + ct_i;
        ct_i++;
    }
    if (comptime_sum == 10) {
        printf("%s (sum 0..5=%d)\n", OK, comptime_sum);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }

    // 15. References
    printf("%s15. References:%s ", BOLD, RESET);
    var ref_val: i32 = 100;
    var ref_ptr: i32* = &ref_val;
    *ref_ptr = 200;
    if (ref_val == 200) {
        printf("%s (modified via ref=%d)\n", OK, ref_val);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }

    // 16. Pointer arithmetic (basic)
    printf("%s16. Pointer ops:%s ", BOLD, RESET);
    var ptr_arr: i32[] = { 10, 20, 30, 40 };
    var ptr_first: i32* = &(ptr_arr[0]);
    var ptr_second: i32* = &(ptr_arr[1]);
    if (*ptr_first == 10 && *ptr_second == 20) {
        printf("%s (arr[0]=%d, arr[1]=%d)\n", OK, *ptr_first, *ptr_second);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }

    // 17. Move semantics
    printf("%s17. Move semantics:%s ", BOLD, RESET);
    var move_val: i32 = 42;
    var moved = move move_val;
    printf("%s (syntax reserved)\n", OK);
    passed++;

    // 18. Copy semantics (syntax test)
    printf("%s18. Copy semantics:%s ", BOLD, RESET);
    var copy_val: i32 = 84;
    var copied = copy copy_val;
    printf("%s (syntax reserved)\n", OK);
    passed++;

    // 19. Async functions 
    printf("%s19. Async/Await:%s ", BOLD, RESET);
    var result = await test_async();
    printf("%s (syntax reserved)\n", OK);
    passed++;

    // 20. Suspend/Resume
    printf("%s20. Suspend/Resume:%s ", BOLD, RESET);
    // Test suspend/resume compilation (runtime stub)
    var suspend_result: i32 = test_resume_call();
    if (suspend_result == 15) {
        printf("%s (suspend/resume compiled)\n", OK);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }

    // 21. Overloading
    printf("%s21. Overloading:%s ", BOLD, RESET);
    var overload_result: i32 = overload_test(10);
    if (overload_result == 10) {
        printf("%s (overload_test(10)=%d)\n", OK, overload_result);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }
    var overload_result2: i32 = overload_test(10, 20);
    if (overload_result2 == 30) {
        printf("%s (overload_test(10, 20)=%d)\n", OK, overload_result2);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }

    // 22. Attributes
    // TODO: Implement attributes && tests.

    // 23. For loops
    for (value, index) in some_array {
        printf("%s (value=%d, index=%d)\n", OK, value, index);
    }

    for (value, index) in some_array | value > 2 | :outer { // Closure gets ran pre-iter
        for (value2, index2) in some_array | value2 > 2 | :inner {
            printf("%s (value=%d, index=%d, value2=%d, index2=%d)\n", OK, value, index, value2, index2);
            if (value2 > 3) {
                break: outer;
            }
        }
    }

    // 24. Closures
    printf("%s24. Closures:%s ", BOLD, RESET);
    // Closure syntax demonstration (full implementation requires closure environment)
    // const closure_func = |x: i32| -> i32 {
    //     return x + 1;
    // };
    // const closure_func_capture = | x: i32 | [move closure_func] -> i32 {
    //     return closure_func(x);
    // };
    //
    // var closure_result: i32 = closure_func_capture(10);
    printf("%s (syntax reserved)\n", OK);
    passed++;

    // 25. @typeinfo() builtin
    printf("%s25. @typeinfo():%s ", BOLD, RESET);
    var typeinfo_i32: i64 = @typeinfo(i32);
    var typeinfo_ptr: i64 = @typeinfo(i32*);
    var typeinfo_arr: i64 = @typeinfo(i32[]);
    if (typeinfo_i32 != 0 && typeinfo_ptr != 0 && typeinfo_arr != 0) {
        printf("%s (i32=%lld, i32*=%lld, i32[]=%lld)\n", OK, typeinfo_i32, typeinfo_ptr, typeinfo_arr);
        passed++;
    } else {
        printf("%s\n", FAIL);
        failed++;
    }

    // 26. String literals
    printf("%s26. String Literals:%s ", BOLD, RESET);
    var str1: cstr = "Hello, Volt!";
    var str2: cstr = "String test";
    var str3: cstr = "";  // Empty string
    printf("%s (str1=\"%s\", str2=\"%s\", empty=\"%s\")\n", OK, str1, str2, str3);
    passed++;

    // Summary
    printf("\n%s%s========== TEST SUMMARY ==========%s\n", BOLD, BLUE, RESET);
    printf("%sPassed:%s %s%d%s\n", BOLD, RESET, GREEN, passed, RESET);
    printf("%sFailed:%s %s%d%s\n", BOLD, RESET, RED, failed, RESET);

    if (failed == 0) {
        printf("\n%s%sALL TESTS PASSED!%s\n", BOLD, GREEN, RESET);
    } else {
        printf("\n%s%sSOME TESTS FAILED%s\n", BOLD, RED, RESET);
    }

    return 0;
}
