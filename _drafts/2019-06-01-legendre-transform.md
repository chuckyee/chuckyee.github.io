---
published: false
---
## Understanding the Legendre Transform, Good Grief

The Legendre transform is one of those mathematical techniques which physics textbooks pull out for theoretical mechanics and thermodynamics which has constantly baffled me. Maybe you've seen in the reformulation of Lagrangian mechanics to create Hamiltonian mechanics. Or you've seen it used to transform the internal energy into the free energy, enthalpy or Gibb's free energy.

The basic idea is to rewrite a function $$F(x)$$ with the independent variable $$x$$ replaced by the derivative $$p(x) = dF(x)/dx$$. You want to do this without any loss of information, meaning you must be able to take your new function $$G(p)$$ and reconstruct $$F(x)$$.

The naive way would be to simply invert the relationship for $$p$$ to get $$x(p)$$ and plug it into the formula for $$F$$, giving $$F(x(p))$$. However, this approach results in a loss of information. Let's work out an example.

Suppose $$F(x) = x^2 + 1$$.