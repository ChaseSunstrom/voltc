
// --- stdlike extern
extern printf(fmt: charptr) -> i32;

// -- math
fn add(a: i32, b: i32) -> i32 { 
    var x = a + b;
    return x;
}

// namespace demo
namespace util::fmt {
    fn one() -> i32 { return 1; }
    const NL: charptr = "\n";
}

use util; // import util::*

fn main() -> i32 {
    var x: i32 = add(2, 3);
    var y = (x > 3) && true || false;
    printf("x=%d y=%d", x as i32, y as i32);
    printf(util::fmt::NL);
    return 0;
}
