# [Deferred Expressions and Contexts](@id defexpr)
## Deferred Expressions
Consider the simple FODO cell:

```@example defexpr
using SciBmad # hide
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  d = Drift(L=1.2)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
end

fodo = Beamline([qf, d, qd, d], E_ref=18e9,
        species_ref=Species("electron"))
```

We set `qf.Kn1 = 0.36`, and `qd.Kn1 = -0.36`. But what if we want to ensure that `qd.Kn1 == -qf.Kn1` always? We can bake-in such an interdependence, common in accelerator parameters, using a "deferred expression" - an expression where evaluation is postponed until its result is actually needed, rather than immediately when it is defined. 

To do this, let's first define a function that returns the current value of `-qf.Kn1`. We can do this without giving the function any explicit name using [lambda/anonymous functions](https://docs.julialang.org/en/v1/manual/functions/#man-anonymous-functions):


```@example defexpr
lambdafun = () -> -qf.Kn1
println("Before: ", lambdafun())
qf.Kn1 = 0.1
println("After: ", lambdafun())
```

Here `lambdafun` takes no arguments (specified by the empty tuple `()`) and returns `-qf.Kn1`. In the context of programming, `lambdafun` is specifically called a [**closure**](https://en.wikipedia.org/wiki/Closure_(computer_programming)), because it "encloses" `qf`, and at the time of evaluation gets the `Kn1` of that enclosed `qf` and negates its sign.

Now we just wrap this function in SciBmad's `DefExpr` type, and we can set any `LineElement` parameter to be such a deferred expression:

```@example defexpr
qd.Kn1 = DefExpr(lambdafun)
qd.Kn1
```

Now if we change `qf.Kn1`, evaluation of `qd.Kn1` will always be `-qf.Kn1`:

```@example defexpr
qf.Kn1 = 0.7
qd.Kn1
```

Deferred expressions can also be manipulated like any other number:

```@example
using SciBmad # hide
a = 1
da = DefExpr(()->a)
b = 2
db = DefExpr(()->b)
dc = da + db
println(dc())
a = 4
println(dc())
dd = sin(dc)
println(dd())
```

One can really "go crazy" with deferred expressions if they want to. They can be infinitely nested, and you can write any function that the programming language allows, for example file I/O, or even control system gets/puts with a real accelerator for a digital twin.

```@docs; canonical=false
DefExpr
```

## Contexts

While `DefExpr`s can wrap variables in the given scope as shown in the previous section, it can be useful and convenient to have a contained place where all control variables exist; this is the purpose of the `Context`. `Context`s contain variables that can be optionally used when evaluating `DefExpr`s that are defined with a single input argument of type `Context`. This is best shown with an example:

```@example context1
using SciBmad # hide
c1 = Context(a = 1);
c2 = Context(a = 2);
d = DefExpr(c -> c.a); # one-argument lambda function
println(d(c1))
println(d(c2))
c1.a = 3; # Can mutate the state of the variables
println(d(c1))
```

Contexts can be pushed on/popped from a global stack of contexts `GLOBAL_CONTEXTS`. In this case, when referencing a variable from a context, if it does not exist in that given context, then the first instance of that variable from the top of the `GLOBAL_CONTEXTS` stack will be used:

```@example context2
using SciBmad # hide
c1 = Context(a = 1);
push!(GLOBAL_CONTEXTS, c1);
c2 = Context(b = 2);
println(c2.a) # `a` does not in exist in `c2`, get from `GLOBAL_CONTEXTS`
push!(GLOBAL_CONTEXTS, c2); 
c3 = Context();
println(c3.a)
println(c3.b)
```

If no context is provided, the first instance of the variable in the `GLOBAL_CONTEXTS` stack will be used:

```@example context3
using SciBmad # hide
c1 = Context(a = 1);
d = DefExpr(c -> c.a); # one-argument lambda function
push!(GLOBAL_CONTEXTS, c1);
println(d()) # Finds `a` from the `GLOBAL_CONTEXTS` stack
```

All `Beamline`s have a `context` property to store a context, which all containing `LineElement` parameters defined use when getting properties at the element-level:

```@example context4
using SciBmad # hide
c1 = Context(Kn1=0.36);
qf = Quadrupole(Kn1=DefExpr(c -> c.Kn1), L=0.5);
bl = Beamline([qf], context=c1);
bl[qf][1].Kn1 # Index the beamline with the `qf` to get all child `qf`s
```
```@docs; canonical=false
Context
```
