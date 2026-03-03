use sesoko::yolo_model::Multiples;

#[test]
fn multiples_all_variants_constructed() {
    for m in [
        Multiples::n(),
        Multiples::s(),
        Multiples::m(),
        Multiples::l(),
        Multiples::x(),
    ] {
        let (f1, f2, f3) = m.filters();
        assert!(f1 > 0, "f1 must be > 0");
        assert!(f2 > 0, "f2 must be > 0");
        assert!(f3 > 0, "f3 must be > 0");
        // Each stage must be wider than or equal to the previous one
        assert!(f2 >= f1, "f2 must be >= f1");
    }
}

#[test]
fn multiples_equality() {
    assert_eq!(Multiples::n(), Multiples::n());
    assert_eq!(Multiples::x(), Multiples::x());
    assert_ne!(Multiples::n(), Multiples::s());
    assert_ne!(Multiples::l(), Multiples::x());
}
