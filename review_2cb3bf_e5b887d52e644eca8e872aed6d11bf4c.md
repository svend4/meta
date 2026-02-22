# Review: 2cb3bf_e5b887d52e644eca8e872aed6d11bf4c.pdf

## Document Information

- **Title:** Введение в теорию касательных сфер (Introduction to the Theory of Tangent Circles/Spheres)
- **Author:** Франц Герман (Franz German), franz.h-n@yandex.ru
- **Pages:** 4 (preview; full work submitted to the Russian Academy of Sciences)

---

## Summary

The paper introduces a **Theory of Center Curves (TCC)** as an alternative geometric framework to the classical **Theory of Conic Sections (TCS)**. The central result is that the locus of centers of circles tangent to two given non-intersecting circles is always a conic section — specifically a hyperbola.

---

## Key Mathematical Content

### Setup

Given two non-intersecting circles, a third tangent circle can be constructed in four distinct configurations:

1. Externally tangent to both given circles
2. Internally tangent to both given circles
3. Externally tangent to one, internally tangent to the other
4. The reverse of case 3

Infinitely many tangent circles exist in each case.

### Coordinate System

- Centers of the two given circles at O and C, with distance OC = n
- Radius of the first circle: r (centered at O)
- Radius of the second circle: R (centered at C)
- M(x, y): center of a tangent circle (the sought locus point)

### Case 1: External Tangency to Both Circles

From the tangency conditions:

```
sqrt(x² + y²) = r + AM
AM = sqrt((n-x)² + y²) - R
```

The locus equation becomes:

**Equation (1):**
```
R - r = sqrt((n-x)² + y²) - sqrt(x² + y²)
```

This is the equation of a **hyperbola** (difference of distances from M to the two foci O and C equals the constant R - r). The locus is the **left branch** of the hyperbola.

With standard substitutions (n = 2c, R - r = 2a, b² = c² - a²) and a coordinate shift, this reduces to the canonical form:

**Equation (2) — Canonical hyperbola:**
```
x*²/a² - y*²/b² = 1
```

### Case 2: Internal Tangency to Both Circles

The analogous derivation yields:

**Equation (3):**
```
R - r = sqrt(x² + y²) - sqrt((n-x)² + y²)
```

This is the **right branch** of the same hyperbola (2).

### Combined General Equation

Cases 1 and 2 together give:

**Equation (4):**
```
R - r = ± sqrt(x² + y²) ∓ sqrt((n-x)² + y²)
```

This represents the full hyperbola with foci at O and C.

### Case 3/4: Mixed Tangency

The paper begins the treatment of the mixed case (externally tangent to one circle, internally tangent to the other) but the visible content ends here, as the full work was submitted to the Russian Academy of Sciences.

---

## Results Summary

| Case | Tangency Type | Resulting Locus |
|------|--------------|-----------------|
| 1    | Both external | Left branch of hyperbola |
| 2    | Both internal | Right branch of hyperbola |
| 1 + 2 | Both | Full hyperbola (Eq. 4) |
| 3/4  | Mixed | Not shown in this excerpt |

---

## Assessment

The paper presents a clean and accessible derivation showing that classical conic sections arise naturally from the geometry of tangent circles. The derivation is straightforward: the defining property of a hyperbola (constant difference of distances to two foci) follows directly from the tangency conditions.

The epigraph citing Coxeter and Greitzer ("These curves can be defined in many different ways") is apt — the paper demonstrates one such alternative definition.

The document is a partial preview; the complete treatment covering all four tangency cases was submitted to the Russian Academy of Sciences.
