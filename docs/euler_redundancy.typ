#set heading(numbering: "1.a.")

#title("Redundancy of Euler Invariant Check in 2D Lee94 Thinning")

= Background

Lee94 preserves topology through four deletion conditions:

+ *Border point*: The candidate must lie on the current surface boundary
+ *Endpoint*: Must not remove the point if it has exactly one foreground neighbor
+ *Euler invariant*: Removal must not change the Euler characteristic
+ *Simple point*: Removal must not disconnect the foreground object

In 3D, the latter two conditions are complementary: the Euler invariant detects creation or destruction of holes and tunnels, while the simple point test detects disconnection of the foreground object.

= 2D Embedding

We can apply the algorithm to 2D images by embedding the input image $I in {0,1}^(H times W)$ into a 3D volume $V in {0,1}^(3 times (H+2) times (W+2))$ as:

$
  V_(p,r,c) = cases(
    I_(r-1,c-1) "if" p = 1,
    0 "otherwise"
  )
$

All neighbors in the $p-1$ and $p+1$ planes are identically zero, collapsing the 26-neighborhood to an effective 8-neighborhood in the $p=1$ plane.

== Border Point Constraint

In the 3D algorithm, a point $x$ is a border point if at least one of its 6-neighbors is background. Due to the embedding where planes $p=0$ and $p=2$ are entirely background ($V=0$), every foreground pixel at $(1, r, c)$ has background neighbors $(0, r, c)$ and $(2, r, c)$. Consequently, _every foreground pixel is a border point_ by the algorithm's definition.

This means the Border Point Constraint does not restrict deletion to the 2D boundary of the shape. Instead, it treats interior pixels of a 2D object as "surface" points of a 1-voxel-thick 3D object. 
The effective check for whether a pixel lies on the 2D boundary must therefore examine only the in-plane 4-neighbors:

$
  exists y in N_4(x) : V_y = 0
$

where $N_4(x) = {(1, r-1, c), (1, r+1, c), (1, r, c-1), (1, r, c+1)}$. The $p$-direction neighbors provide no constraining power since they are always background.

= Impossibility of Hole Creation

*Lemma 1*: _In 2D Lee94 thinning, no border point removal can create a hole ($delta H > 0$)._

*Proof.* For a hole to be created by removing $x$, the 4-neighbors of $x$ must all be foreground. Otherwise, if any 4-neighbor is background, it provides a 4-connected path from the newly-background $x$ to the exterior, so no new isolated background component (hole) is formed:

$
  exists y in N_4(x) : V_y = 0 arrow "no hole created"
$

However, if all 4-neighbors are foreground:

$
  forall y in N_4(x) : V_y = 1
$

then $x$ fails the border point condition and is never a candidate for deletion. Therefore, no border point removal can create a hole. $square$

= Impossibility of Hole Elimination

*Lemma 2*: _In 2D Lee94 thinning, no simple point removal can eliminate a hole ($delta H < 0$)._

*Proof.* A hole is eliminated when its interior background component merges with the exterior background. For this to occur via removal of $x$, the pixel $x$ must be the sole separator between the hole interior and the exterior.

In the local neighborhood $N(x)$, this means there exist at least two disconnected background components: one belonging to the hole interior and one belonging to the exterior. For $x$ to be the only barrier between them, the foreground pixels in $N(x)$ must form at least two disconnected components that "wrap around" the separate background regions.

By definition, a point is simple only if the number of connected foreground components in its neighborhood satisfies $O(S inter N(x)) = 1$. If removal of $x$ would eliminate a hole, then $O(S inter N(x)) >= 2$, and $x$ fails the Simple-Point-Check.

Therefore, no simple point removal can eliminate a hole. $square$

= Equivalence of Topology Checks

*Theorem*: _For 2D Lee94 thinning, the Euler invariant check is redundant._

*Proof.* The Euler characteristic $chi$ for a 2D binary image is:

$
  chi = O - H
$

where $O$ is the number of foreground connected components and $H$ is the number of holes.

When removing a pixel $x$ that is both a border point and a simple point:
+ $delta H > 0$ (hole creation) is impossible by Lemma 1 (border point condition)
+ $delta H < 0$ (hole elimination) is impossible by Lemma 2 (simple point condition)
+ Therefore $delta H = 0$

This reduces the Euler characteristic change to:

$
  delta chi = delta O - delta H = delta O
$

The Simple-Point-Check ensures $delta O = 0$ (no disconnection). Since $delta chi = delta O$ for all valid deletion candidates, the Euler-Invariant-Check ($delta chi = 0$) is automatically satisfied whenever the Simple-Point-Check passes. $square$

= Practical Implications and Considerations

For 2D thinning applications:
+ No 3D embedding is required --- the algorithm can be adapted to operate directly on a padded 2D image $I in {0,1}^((H+2) times (W+2))$
+ Border point detection simplifies from 6-neighbor 3D checks to 4-neighbor 2D checks: a pixel is a border point if any of its 4-connected neighbors (N, S, E, W) is background
+ The Euler-Invariant-Check can be omitted, the Simple-Point-Check alone suffices for topology preservation
+ Simple-Point-Check can be computed using a direct flood fill: collecting all 8-connected foreground neighbors and running DFS to count connected components, avoiding the building and traversal of an octree
