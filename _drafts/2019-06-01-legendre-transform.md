---
layout: post
title: "Legendre Transforms: Good Grief"
published: true
mathjax: true
---

The Legendre transform is one of those mathematical techniques which physics
textbooks pull out with little explanation and intuition. Most physics students
encounter it in classical mechanics, where it's used to connect the Lagrangian
$$L(\dot{q}, q)$$ to the Hamiltonian $$H(p, q)$$, and in thermodynamics, where
it yields the relation between the internal energy $$E$$ and other
thermodynamic potentials like the free energy or enthalpy.

But why is the Legendre transform $$G(p)$$ of a function $$F(x)$$ defined by
$$G \equiv px - F$$? How exactly is $$p$$ the derivative of $$F$$? What even is
the independent variable here? And if you got all that, could someone explain
the geometric intuition behind the Legendre transform?

[//]: Motivate the post by appealing to frustration and common misconceptions of Legendre transforms
[//]: Have a picture as early on as possible

## Let's start with geometry

Lines to be precise. Remember this formula for a line with slope $$a$$ and
$$y$$-intercept $$b$$?

$$y = ax + b$$

Here is the first key idea: each line in the 2d plane can be uniquely
identified by a pair of numbers $$(a, b)$$. For example, the point $$(1, 0)$$
in the $$a$$-$$b$$ plane represents the line $$y = x$$ in the $$x$$-$$y$$ plane:

![](/images/legendre/abxy1.png)

The point $$(a, b) = (0, 1)$$ represents the line $$y = 1$$:

![](/images/legendre/abxy2.png)

If we consider the set of points forming an arc, then we get a bundle of lines:

![](/images/legendre/abxy-arc.png)

Examining the bundle of lines, you may notice that they behave like the set of
tangent lines to a curve. Maybe the arc in $$a$$-$$b$$ space relates to the
curve in $$x$$-$$y$$ space in some way.

![](/images/legendre/abxy-curves.png)

That relationship can be made precise: we've wandered upon the idea of duality.

## Tangent lines and duality

Let's formalize the idea of duality. 

Tangent lines to be precise. Suppose you have a convex function (the reason for
the restriction will become clear in a moment). Here's one:

![F(x)](/images/legendre/fx.png)

The legendre transform is so fundamental that it is one of those ideas which
have many viewpoints, each of which highlights one aspect of the idea.

* Geometric: areas, supporting hyperplanes
* Units of conjugate variables

The basic idea is to rewrite a function $$F(x)$$ with the independent variable $$x$$ replaced by the derivative $$p(x) = dF(x)/dx$$. You want to do this without any loss of information, meaning you must be able to take your new function $$G(p)$$ and reconstruct $$F(x)$$.

The naive way would be to simply invert the relationship for $$p$$ to get $$x(p)$$ and plug it into the formula for $$F$$, giving $$F(x(p))$$. However, this approach results in a loss of information.

Let's work out an example in 1d: suppose our function is $$F(x) = x^2 + 1$$.

[Explain the mathematics of the transform, provide an intuition of how it works]

![area-interpretation](/images/legendre/area.png)

[Provide sample mathematical examples]

[Provide explanation of how they are used in physics and the physical meaning of the Legendre transform]


## Resources

https://ps.uci.edu/~cyu/p115B/class.html

https://doi.org/10.1119/1.3119512

https://www.aapt.org/docdirectory/meetingpresentations/SM14/Mungan-Poster.pdf

https://physics.stackexchange.com/questions/4384/physical-meaning-of-legendre-transformation

https://jmanton.wordpress.com/2010/11/21/introduction-to-the-legendre-transform/

https://www.mia.uni-saarland.de/Teaching/NAIA07/naia07_h3_slides.pdf

https://en.wikipedia.org/wiki/Supporting_hyperplane

https://www.andrew.cmu.edu/course/33-765/pdf/Legendre.pdf