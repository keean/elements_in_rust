#![feature(unboxed_closures)]

pub mod elements {

    extern crate num;

    use std::ops::{Add, Sub, Shr};
    use self::num::{Zero, One, NumCast};

    //-----------------------------------------------------------------------------
    // 1.5 Regular Types

    // Regular types enable normal value semantics, except we require explicit
    // cloning rather than copying so the cost is visible in the code.
    pub trait Regular : PartialEq + Clone {}

    // A type is regular if it has assignment, equality, and is clonable.
    impl<I> Regular for I where I : PartialEq + Clone {}

    //-----------------------------------------------------------------------------
    // 2.1 Integers

    // This is an incomplete implementation of the Integer concept with just the 
    // functions necessary for the iterator algorithms below.
    pub trait Integer : num::Integer 
    where Self : Regular + Shr<Self, Output = Self> 
    {
        fn two() -> Self where Self : NumCast {
            Self::from(2).unwrap()
        }
        fn is_two(&self) -> bool where Self : NumCast + PartialEq {
            *self == Self::from(2).unwrap()
        }
        fn successor(self) -> Self where Self : Sized {
            self + Self::one()
        }
        fn predecessor(self) -> Self where Self : Sized {
            self - Self::one()
        }
        fn half_nonnegative(self) -> Self where Self : Sized {
            self >> Self::one()
        }
    }

    impl<I> Integer for I
    where I : num::Integer + Regular + Shr<I, Output = I> {}

    //-----------------------------------------------------------------------------
    // 6.1 Readability

    // Readability is roughly equivalent to a read-only reference in Rust, but also
    // represents read-only access to containers. Ideally this is a read-only
    // version of Rusts 'Deref' trait, but we would like a value to derefence to
    // itself.
    pub trait Readable {
        type ValueType : Regular;
        fn source(&self) -> &Self::ValueType;
    }

    //-----------------------------------------------------------------------------
    // 6.2 Iterators

    // A plain Iterator is one which can only pass through the data once. To
    // enforce this it must not be copyable, and successor needs to consume the
    // current iterator to return a new one. This means that we define increment
    // which mutates the iterator instead of successor, as we can then implement
    // successor without copying the iterator or moving it out of the borrowed
    // context.
    pub trait IteratorImpl : PartialEq {
        type DistanceTypeImpl : Integer;
        fn increment_impl(&mut self);
    }       

    // Note: We need a wrapper for iterators, so that we can define addition and
    // subtraction using the normal operators.
    #[derive(Clone, PartialEq, Debug)]
    pub struct It<I>(pub I);

    pub trait Iterator : PartialEq
    where Self : Sized
        + Add<<Self as Iterator>::DistanceType, Output = Self>
        + Sub<Self, Output = <Self as Iterator>::DistanceType>
    {
        type DistanceType : Integer;
        fn increment(&mut self);
        fn successor(mut self) -> Self where Self : Sized {
            self.increment();
            self
        }
    }       

    impl<I> Readable for It<I> where I : Readable {
        type ValueType = I::ValueType;
        fn source(&self) -> &Self::ValueType {
            self.0.source()
        }
    }

    impl<I> Iterator for It<I>
    where I : IteratorImpl {
        type DistanceType = I::DistanceTypeImpl;
        fn increment(&mut self) {
            self.0.increment_impl();
        }
    }

    //-----------------------------------------------------------------------------
    // 6.3 Ranges

    impl<I> Add<I::DistanceTypeImpl> for It<I> where I : IteratorImpl {
        type Output = Self;
        fn add(mut self, mut n : I::DistanceTypeImpl) -> Self {
            // Precondition: n >= 0 && weak_range(f, n)
            while n != I::DistanceTypeImpl::zero() {
                n = n.predecessor();
                self = self.successor();
            }
            self
        }
    }

    impl<I> Sub<It<I>> for It<I> where I : IteratorImpl {
        type Output = I::DistanceTypeImpl;
        fn sub(self, mut f : Self) -> Self::Output {
            // Precondition: bounded_range(f, l)
            let mut n = I::DistanceTypeImpl::zero();
            while f != self {  
                n = n.successor();
                f = f.successor();
            }
            n
        }
    }

    //-----------------------------------------------------------------------------
    // 6.4 Readable Ranges

    pub fn for_each<I, P>(mut f : I, l : &I, mut p : P) -> P
    where I : Iterator + Readable, P : FnMut(&I::ValueType) {
        // Precondition: readable_bounded_range(f, l)
        while f != *l {
            p(f.source());
            f = f.successor();
        }
        p
    }

    pub fn find<I>(mut f : I, l : &I, x : I::ValueType) -> I
    where I : Iterator + Readable {
        // Precondition: readable_bounded_range(f, l)
        while (f != *l) && (*(f.source()) != x) {
            f = f.successor();
        }
        f 
    }
    
    pub fn find_if<I, P>(mut f : I, l : &I, mut p : P) -> I
    where I : Iterator + Readable, P : FnMut(&I::ValueType) -> bool {
        // Precondition: readable_bounded_range(f, l)
        while (f != *l) && !p(f.source()) {
            f = f.successor();
        }
        f 
    }

    pub fn find_if_not<I, P>(mut f : I, l : &I, mut p : P) -> I
    where I : Iterator + Readable, P : FnMut(&I::ValueType) -> bool {
        // Precondition: readable_bounded_range(f, l)
        while (f != *l) && p(f.source()) {
            f = f.successor();
        }
        f
    }

    pub fn count_if<I, J, P>(mut f : I, l : &I, mut p : P, mut j : J) -> J
    where I : Iterator + Readable, J : Integer, P : FnMut(&I::ValueType) -> bool {
        // Precondition: readable_bounded_range(f, l)
        while f != *l {
            if p(f.source()) {
                j = j.successor();
            }
            f = f.successor();
        }
        j
    }

    pub fn count_if_from_zero<I, P>(f : I, l : &I, p : P) -> I::DistanceType
    where I : Iterator + Readable, P : FnMut(&I::ValueType) -> bool {
        // Precondition: readable_bounded_range(f, l)
        count_if(f, l, p, I::DistanceType::zero())
    }

    pub fn fold<I, Op>(mut f : I, l : &I, mut op : Op, mut r : I::ValueType) -> I::ValueType
    where I : Iterator + Readable, Op : FnMut(&I::ValueType, &I::ValueType) -> I::ValueType {
        // Precondition: readable_bounded_range(f, l)
        // Precondition: partially_associative(op) 
        while f != *l {
            r = op(&r, f.source());
            f = f.successor();
        }
        r
    }

    pub fn reduce_nonempty<I, Op, F, D>(mut f : I, l : &I, mut op : Op, mut fun : F) -> D 
    where I : Iterator + Readable, Op : FnMut(D, D) -> D, F : FnMut(&I) -> D {
        // Precondition: readable_bounded_range(f, l)
        // Precondition: partially_associative(op)
        // Precondition: forall x in [f,l) fun(x) is defined 
        let mut r = fun(&f);
        f = f.successor();
        while f != *l {
            r = op(r, fun(&f));
            f = f.successor();
        }
        r
    }

    pub fn reduce<I, Op, F, D>(f : I, l : &I, op : Op, fun : F, z : D) -> D
    where D : Regular, I : Iterator + Readable, Op : FnMut(D, D) -> D, F : FnMut(&I) -> D {
        // Precondition: readable_bounded_range(f, l)
        // Precondition: partially_associative(op)
        // Precondition: forall x in [f,l) fun(x) is defined 
        if f == *l {
            z
        } else {
            reduce_nonempty(f, l, op, fun)
        }
    }

    pub fn reduce_nonzeroes<I, Op, F, D>(mut f : I, l : &I, mut op : Op, mut fun : F, z : D) -> D 
    where D : Regular, I : Iterator + Readable, Op : FnMut(D, D) -> D, F : FnMut(&I) -> D {
        // Precondition: readable_bounded_range(f, l)
        // Precondition: partially_associative(op)
        // Precondition: forall x in [f,l) fun(x) is defined 
        let mut x : D;
        while {
            if f == *l {
               return z;
            }
            x = fun(&f);
            f = f.successor();
            x == z
        } {}
        while f != *l {
            let y = fun(&f);
            if y != z {
                x = op(x, y);
            }
            f = f.successor();
        }
        x
    }

    pub fn for_each_n<I, P>(mut f : I, mut n : I::DistanceType, mut p : P) -> (P, I)
    where I : Iterator + Readable, P : FnMut(&I::ValueType) {
        // Precondition: readable_weak_range(f, n)
        while I::DistanceType::zero() != n {
            n = n.predecessor();
            p(f.source());
            f = f.successor();
        }
        (p, f)
    } 

    pub fn find_n<I>(mut f : I, mut n : I::DistanceType, x : I::ValueType) -> (I, I::DistanceType)
    where I : Iterator + Readable {
        // Precondition: weak_range(f, n)
        while n != I::DistanceType::zero() && *(f.source()) != x {
            n = n.predecessor();
            f = f.successor();
        }
        (f, n)
    }

    pub fn find_if_unguarded<I, P>(mut f : I, mut p : P) -> I
    where I : Iterator + Readable, P : FnMut(&I::ValueType) -> bool {
        // Precondition: exists l . readable_bounded_range(f, l) && some(f, j, p)
        while !p(f.source()) {
            f = f.successor();
        }
        f
        // Postcondition: p(f.source())
    }

    pub fn find_mismatch<I0, I1, R, V>(mut f0 : I0, l0 : &I0, mut f1 : I1, l1 : &I1, mut r : R) -> (I0, I1)
    where I0 : Iterator + Readable<ValueType = V>, I1 : Iterator + Readable<ValueType = V>,
    R : FnMut(&V, &V) -> bool, V : Regular {
        // Precondition: readable_bounded_range(f0, l0)
        // Precondition: readable_bounded_range(f1, l1)
        while f0 != *l0 && f1 != *l1 && r(f0.source(), f1.source()) {
            f0 = f0.successor();
            f1 = f1.successor();
        }
        (f0, f1)
    }

    // Note: this algorithm needs to clone the data as the iterator is a single pass 
    // iterator. That means you cannot hold a reference to the location pointed to by
    // 'f' once 'f' has been incremented. To do so would effectively allow the iterator
    // to be copied, which would break the invarient conditions. 
    pub fn find_adjacent_mismatch<I, R>(mut f : I, l : &I, mut r : R) -> I
    where I : Iterator + Readable, R : FnMut(&I::ValueType, &I::ValueType) -> bool {
        // Precondition: readable_bounded_range(f, l)
        if f != *l {
            let mut x : I::ValueType = (*f.source()).clone();
            f = f.successor();
            while (f != *l) && r(&x, f.source()) {
                x = (*f.source()).clone();
                f = f.successor();
            }
        }
        f
    }

    //-----------------------------------------------------------------------------
    // 6.5 Increasing Ranges

    pub fn relation_preserving<I, R>(f : I, l : &I, r : R) -> bool
    where I : Iterator + Readable, R : FnMut(&I::ValueType, &I::ValueType) -> bool {
        // Precondition: readable_bounded_range(f, l)
        *l == find_adjacent_mismatch(f, l, r)
    }

    pub fn strictly_increasing_range<I, R>(f : I, l : &I, r : R) -> bool
    where I : Iterator + Readable, R : FnMut(&I::ValueType, &I::ValueType) -> bool {
        // Precondition: readable_bounded_range(f, l) && weak_ordering(r)
        relation_preserving(f, l, r)
    }

    pub fn complement_of_converse<'a, R, A>(mut r : R) -> Box<FnMut(&A, &A) -> bool + 'a>
    where R : FnMut(&A, &A) -> bool + 'a {
        Box::new(move |a, b| !r(b, a))
    }

    pub fn increasing_range<I, R>(f : I, l : &I, r : R) -> bool
    where I : Iterator + Readable, R : FnMut(&I::ValueType, &I::ValueType) -> bool + Sized {
        // Precondition: readable_bounded_range(f, l) && weak_ordering(r)
        relation_preserving(f, l, &mut *complement_of_converse(r))
    }

    pub fn partitioned<I, P>(f : I, l : &I, mut p : P) -> bool
    where I : Iterator + Readable, P : FnMut(&I::ValueType) -> bool {
        // Precondition: readable_bounded_range(f, l)
        let g = find_if(f, l, &mut p);
        *l == find_if_not(g, l, p)
    }

    //-----------------------------------------------------------------------------
    // 6.6 Forward Iterators

    // A Forward Iterator is an Iterator that is also a Regular type, this means it
    // must behave like a value, and be copyable, assignable (storable) in addition
    // to having an equality operator.
    pub trait ForwardIterator : Iterator + Regular {}

    impl<I> ForwardIterator for It<I> where It<I> : Iterator + Regular {}

    // Because a forward iterator is copyable we cam clone the iterator instead
    // of having to clone the data.
    pub fn find_adjacent_mismatch_forward<I, R>(mut f : I, l : &I, mut r : R) -> I
    where I : ForwardIterator + Readable, R : FnMut(&I::ValueType, &I::ValueType) -> bool {
        // Precondition: readable_bounded_range(f, l)
        if f == *l {
            return f; // return f not l because we own it
        }
        let mut t : I;
        while {
            t = f.clone();
            f = f.successor();
            f != *l && r(t.source(), f.source())
        } {} // would loop/break be better?
        f
    }

    //pub fn partition_point_n<I, P>(mut f : I, mut n : I::DistanceType, mut p : P) -> I
    pub fn partition_point_n<I, P>(mut f : I, mut n : I::DistanceType, mut p : P) -> I
    where I : ForwardIterator + Readable, P : FnMut(&I::ValueType) -> bool {
        // Precondition: readable_counted_range(f, n) && partitioned_n(f, n, p)
        while !n.is_zero() {
            let h : I::DistanceType = n.clone().half_nonnegative();
            let m : I = f.clone() + h.clone();
            if p(m.source()) {
                n = h;
            } else {
                n = n - h.successor();
                f = m.successor();
            }
        }
        f
    }

    pub fn partition_point<I, P>(f : I, l : I, p : P) -> I
    where I : ForwardIterator + Readable, P : FnMut(&I::ValueType) -> bool {
        // Precondition: readable_bounded_range(f, n) && partitioned(f, l, p)
        partition_point_n(f.clone(), l - f, p)
    }

    pub fn lower_bound_predicate<'a, R, D>(a : &'a D, mut r : R) -> Box<FnMut(&D) -> bool + 'a>
    where R : FnMut(&D, &D) -> bool + 'a {
        Box::new(move |x| !r(x, a))
    }

    pub fn lower_bound_n<I, R>(f : I, n : I::DistanceType, a : &I::ValueType, r : R) -> I
    where I : Readable + ForwardIterator, R : FnMut(&I::ValueType, &I::ValueType) -> bool {
        // Precondition: weak_ordering(r) && increasing_counted_range(f, n, r)
        partition_point_n(f, n, &mut *lower_bound_predicate(a, r))
    }

    pub fn upper_bound_predicate<'a, R, D>(a : &'a D, mut r : R) -> Box<FnMut(&D) -> bool + 'a>
    where R : FnMut(&D, &D) -> bool + 'a {
        Box::new(move |x| r(a, x))
    }

    pub fn upper_bound_n<I, R>(f : I, n : I::DistanceType, a : &I::ValueType, r : R) -> I
    where I : Readable + ForwardIterator, R : FnMut(&I::ValueType, &I::ValueType) -> bool {
        // Precondition: weak_ordering(r) && increasing_counted_range(f, n, r)
        partition_point_n(f, n, &mut *upper_bound_predicate(a, r))
    }

    //-----------------------------------------------------------------------------
    // 6.7 Indexed Iterators

    pub trait IndexedIterator : ForwardIterator {
        fn add(self, Self::DistanceType) -> Self;
        fn sub(self, Self) -> Self::DistanceType;
    }

    //-----------------------------------------------------------------------------
    // 6.7 Indexed Iterators

    pub trait BidirectionalIterator : ForwardIterator {
        fn predecessor(self) -> Self;
    }
}

//=============================================================================

#[cfg(test)]
mod test {

    extern crate num;

    use std::fmt::*;
    use std::ops::*;
    use elements::*;

    //-----------------------------------------------------------------------------
    // Define Slice Iterator

    #[derive(Clone, PartialEq, Debug)]
    struct SliceIterator<T> {
        ptr : *mut T
    } 

    impl<T> SliceIterator<T> {
        fn new(r : &mut T) -> It<SliceIterator<T>> {
            It(SliceIterator {
                ptr : r as *mut T
            })
        }
    }

    impl<T> Display for SliceIterator<T> where T : Display {
        fn fmt(&self, f : &mut Formatter) -> Result {
            write!(f, "{}", self.ptr as usize)
        }
    }

    impl<T> Readable for SliceIterator<T> where T : Regular {
        type ValueType = T;
        fn source(&self) -> &T {
            let v : &T;
            unsafe { 
                v = &*((*self).ptr);
            }
            v
        }
    }

    impl <T> IteratorImpl for SliceIterator<T> where T : PartialEq {
        type DistanceTypeImpl = usize;
        fn increment_impl(&mut self) {
            unsafe {
                self.ptr = self.ptr.offset(1);
            }
        }
    }

    //-----------------------------------------------------------------------------
    // Test Slice Iterator
    
    fn test_for_each<I>(f : I, l : &I, j : I::ValueType, k : I::ValueType)
    where I : Readable + Iterator,
    <I as Readable>::ValueType : AddAssign<I::ValueType> + Debug {
        let mut s : I::ValueType = j;
        for_each(f, l, |v| s += (*v).clone());
        assert_eq!(s, k);
    }

    fn test_find<I>(f : I, l : &I, i : I::ValueType, j : I::ValueType, k : I::ValueType)
    where I : Readable + Iterator,
    <I as Readable>::ValueType : AddAssign<I::ValueType> + Debug {
        let mut s : I::ValueType = i;
        for_each(find(f, l, j), l, |v| s += (*v).clone());
        assert_eq!(s, k);
    } 

    fn test_find_if<I>(f : I, l : &I, i : I::ValueType, j : I::ValueType, k : I::ValueType)
    where I : Readable + Iterator,
    <I as Readable>::ValueType : AddAssign<I::ValueType> + Debug { 
        let mut s : I::ValueType = i;
        for_each(find_if(f, l, |v| *v==j), l, |v| s += (*v).clone());
        assert_eq!(s, k);
    }

    fn test_count_if<I>(f : I, l : &I, i : I::ValueType, j : I::ValueType,
        k : I::ValueType, m : I::ValueType)
    where I : Readable + Iterator, I::ValueType : Integer + Debug {
        let c1 : I::ValueType = count_if(f, l, |v| *v > i && *v < j, k);
        assert_eq!(c1, m);
    }

    fn test_count_if_from_zero<I>(f : I, l : &I, i : I::ValueType,
        j : I::ValueType, k : I::DistanceType)
    where I : Readable + Iterator, I::ValueType: PartialOrd, I::DistanceType : Debug {
        let c1 : I::DistanceType = count_if_from_zero(f, l, |v| *v > i && *v < j);
        assert_eq!(c1, k);
    }

    fn test_reduce_nonempty<I>(f : I, l : &I, k : I::ValueType)
    where I : Readable + Iterator,
    <I as Readable>::ValueType: Debug + Add<Output = I::ValueType> {
        let r = reduce_nonempty(f, l, |a, b| a + b, |a| (*a.source()).clone());
        assert_eq!(r, k);
    }

    fn test_reduce<I>(f : I, l : &I, i : I::ValueType, j : I::ValueType)
    where I : Iterator + Readable,
    <I as Readable>::ValueType : Add<Output = I::ValueType> + Debug {
        let r = reduce(f, l, |a, b| a + b, |a| (*a.source()).clone(), i);
        assert_eq!(r, j);
    }

    fn test_reduce_nonzeroes<I>(f : I, l : &I, i : I::ValueType, j : I::ValueType)
    where I : Readable + Iterator,
    <I as Readable>::ValueType : Add<Output = I::ValueType> + Debug {
        let r = reduce_nonzeroes(f, l, |a, b| a + b, |a| (*a.source()).clone(), i);
        assert_eq!(r, j);
    }

    fn test_for_each_n<I>(f : I, n : I::DistanceType, j : I::ValueType, k : I::ValueType)
    where I : Readable + Iterator,
    <I as Readable>::ValueType : AddAssign<I::ValueType> + Debug {
        let mut s : I::ValueType = j;
        for_each_n(f, n, |v| s += (*v).clone());
        assert_eq!(s, k);
    }

    fn test_find_n<I>(f : I, n : I::DistanceType, i : I::ValueType,
        j : I::ValueType, k : I::ValueType)
    where I : Readable + Iterator,
    <I as Readable>::ValueType : AddAssign<I::ValueType> + Debug {
        let mut s : I::ValueType = i;
        let (g, m) = find_n(f, n, j);
        for_each_n(g, m, |v| s += (*v).clone());
        assert_eq!(s, k);
    }

    fn test_find_if_unguarded<I>(f : I, l : &I, i : I::ValueType, j : I::ValueType,
        k : I::ValueType)
    where I : Readable + Iterator,
    <I as Readable>::ValueType : AddAssign<I::ValueType> + Debug {
        let mut s : I::ValueType = i;
        for_each(find_if_unguarded(f, |v| *v==j), l, |v| s += (*v).clone());
        assert_eq!(s, k);
    }

    fn test_find_mismatch<I>(f0 : I, l0 : &I, f1 : I, l1 : &I)
    where I : Readable + Iterator {
        let (i,j) = find_mismatch(f0, l0, f1, l1, |a, b| a == b);
        assert!(*i.source() != *j.source());
    }

    fn test_find_adjacent_mismatch<I>(f : I, l : &I, m : &I)
    where I : Readable + Iterator, I::ValueType : Debug {
        let i = find_adjacent_mismatch(f, l, |a, b| a == b);
        assert!(i.source() != (*m).source());
    }

    fn test_relation_preserving<I>(f : I, l : &I)
    where I : Readable + Iterator, I::ValueType : PartialOrd {
        let b = relation_preserving(f, l, |a, b| b > a);
        assert!(b);
    }

    fn test_strictly_increasing_range<I>(f : I, l : &I)
    where I : Readable + Iterator, I::ValueType : PartialOrd {
        let b = strictly_increasing_range(f, l, |a, b| b > a);
        assert!(b);
    }

    fn test_complement_of_converse() {
        let mut r = complement_of_converse(|a, b| b > a);
        assert!(r(&3, &4) == true);
        assert!(r(&4, &4) == true);
        assert!(r(&5, &4) == false);
    }

    fn test_increasing_range<I>(f : I, l : &I)
    where I : Readable + Iterator, I::ValueType : PartialOrd {
        let b = increasing_range(f, l, |a, b| b > a);
        assert!(b);
    }

    fn test_partitioned<I>(f : I, l : &I, x : I::ValueType) 
    where I : Readable + Iterator, I::ValueType : PartialOrd {
        let b = partitioned(f, l, |a| a > &x);
        assert!(b);
    }

    fn test_find_adjacent_mismatch_forward<I>(f : I, l : &I, m : &I)
    where I : Readable + ForwardIterator {
        let i = find_adjacent_mismatch_forward(f, l, |a, b| a == b);
        assert!(i.source() != (*m).source());
    }

    fn test_partition_point<I>(f : I, l : &I, p : I::ValueType, q : I::DistanceType)
    where I : Readable + ForwardIterator, I::DistanceType : PartialEq {
        let i = partition_point(f.clone(), l.clone(), |a| a == &p);
        assert!(l.clone() - i == q);
    }

    fn test_lower_bound_n<I>(f : I, n : I::DistanceType, x : I::ValueType, y : I::ValueType)
    where I : Readable + ForwardIterator, I::ValueType : PartialOrd,
    I::DistanceType : PartialEq {
        let i = lower_bound_n(f.clone(), n, &x, |a, b| a < b);
        assert!(i.source() == &y);
    }

    fn test_upper_bound_n<I>(f : I, n : I::DistanceType, x : I::ValueType, y : I::ValueType)
    where I : Readable + ForwardIterator, I::ValueType : PartialOrd,
    I::DistanceType : PartialEq {
        let i = upper_bound_n(f.clone(), n, &x, |a, b| a < b);
        assert!(i.source() == &y);
    }

    #[test]
    fn test_iterators() {
        let mut v = [0, 1, 2, 3];
        let mut w = [0, 1, 3, 2];
        let f = SliceIterator::new(v.first_mut().unwrap());
        let g = SliceIterator::new(w.first_mut().unwrap());
        let l = f.clone() + v.len();
        let m = g.clone() + w.len();
        assert_eq!(v.len(), l.clone() - f.clone());
        assert_eq!(w.len(), m.clone() - g.clone());

        test_for_each(f.clone(), &l, 0, 6);
        test_find(f.clone(), &l, 0, 2, 5);
        test_find_if(f.clone(), &l, 0, 1, 6);
        test_count_if(f.clone(), &l, 0, 3, 1, 3);
        test_count_if_from_zero(f.clone(), &l, 0, 3, 2);
        test_reduce_nonempty(f.clone(), &l, 6);
        test_reduce(f.clone(), &l, 0, 6);
        test_reduce(f.clone(), &f, 0, 0);
        test_reduce_nonzeroes(f.clone(), &l, 0, 6);
        test_for_each_n(f.clone(), v.len(), 0, 6);
        test_find_n(f.clone(), v.len(), 0, 2, 5);
        test_find_if_unguarded(f.clone(), &l, 0, 2, 5);
        test_find_mismatch(f.clone(), &l, g.clone(), &m);
        test_find_adjacent_mismatch(f.clone(), &l, &f);
        test_relation_preserving(f.clone(), &l);
        test_strictly_increasing_range(f.clone(), &l);
        test_complement_of_converse();
        test_increasing_range(f.clone(), &l);
        test_partitioned(f.clone(), &l, 2);
        test_find_adjacent_mismatch_forward(f.clone(), &l, &f);
        test_partition_point(f.clone(), &l, 2, 2);
        test_lower_bound_n(f.clone(), v.len(), 1, 1);
        test_upper_bound_n(f.clone(), v.len(), 1, 2);
    }
}

