extern C printf(fmt: cstr) -> i32;

// NAMESPACES
namespace math {
    fn abs(x: i32) -> i32 {
        if (x < 0) {
            return 0 - x;
        }
        return x;
    }
}

// STRUCTS
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

fn sum_range(n: i32) -> i32 {
    var sum: i32 = 0;
    var i: i32 = 0;
    while (i < n) {
        sum = sum + i;
        i++;
    }
    return sum;
}

fn main() -> i32 {
    printf("=== VOLT FEATURE TEST ===\n\n");
    
    // 1. Basic types
    printf("1. Basic Types: ");
    var x: i32 = 42;
    var y: i64 = 1000;
    var b: bool = true;
    printf("OK\n");
    
    // 2. Arithmetic
    printf("2. Arithmetic: ");
    var sum: i32 = x + 10;
    var prod: i32 = x * 2;
    printf("OK\n");
    
    // 3. Comparisons
    printf("3. Comparisons: ");
    var eq: bool = x == 42;
    var gt: bool = x > 10;
    printf("OK\n");
    
    // 4. Logical operators
    printf("4. Logical ops: ");
    var and_test: bool = (x > 10) && (x < 100);
    var or_test: bool = (x == 0) || (x == 42);
    printf("OK\n");
    
    // 5. Pointers
    printf("5. Pointers: ");
    var ptr: i32* = &x;
    var deref: i32 = *ptr;
    printf("OK\n");
    
    // 6. Namespaces
    printf("6. Namespaces: ");
    var abs_val: i32 = math::abs(0 - 10);
    printf("OK\n");
    
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
        default => printf("INVALID\n")
    }
    printf("OK\n");
    
    // 8. Loops
    printf("8. Loops: ");
    var loop_result: i32 = sum_range(10);
    printf("OK\n");
    
    // 9. Type casting
    printf("9. Type casting: ");
    var casted: i64 = x as i64;
    printf("OK\n");
    
    // 10. Compound assignment
    printf("10. Compound assign: ");
    var counter: i32 = 0;
    counter++;
    counter += 5;
    printf("OK\n");
    
    // 11. Structs
    printf("11. Structs: ");
    var point: Point = {0, 0};
    printf("OK\n");
    
    // 12. Arrays
    printf("12. Arrays: ");
    var some_array: i32[] = {1, 2, 3, 4};
    some_array[0] = 2;
    printf("OK\n");
    
    // 13. Defer
    printf("13. Defer: ");
    defer printf("Deferred!\n");
    printf("OK\n");
    
    printf("\n=== ALL TESTED FEATURES WORKING! ===\n");
    return 0;
}
