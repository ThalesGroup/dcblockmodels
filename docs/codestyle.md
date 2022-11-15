# Code style specifics

This codebase strives to follow PEP8 standards, but some specifics have been needed for code readability and easier links to the articles describing the algorithm.

## Variable names

Since a lot of long equations are implemented, shortened variable names need to be accepted. Also, the maths involve a lot of upper case variables that describe matrices, functions, which cannot be lower case'd without loss of understandability in the code. Therefore, there is some tolerance to the following variables:

`ND`, `Kz`, `Kw`, `nax`, `ax`, `N`, `D`, `N_`, `D_`, `T`, `G`, `H`, `a`, `c`, `d`, `f`, `df`, `ix`, `ca`, `g1`, `g2`, `i`, `j`, `k`, `k_`, `l`, `n`, `n1`, `n2`, `p`, `s`, `t`, `t_`, `u`, `y`, `it`, `mu`, `nu`, `pi`, `p1`, `p2`, `L1`, `L2`, `dLBM`, `L`, `N_`, `V`, `B`, `D_`, `Hz`, `Hw`, `qw`, `qz`, `root_D_inv`, `x`, `x0`, `x1`, `R`, `C`, `CA_plot`

and any variables that respect the following regular expressions:

`Z.*`, `W.*`, `.*Z`, `.*W`, `.*X.*`, `K.*`, `.*S.*`, `.*P.*`, `.*Lc.*`, `.*ND`, `AFD.*`, `AR.*`

## Code structure

Some leeway has been left to line length for modules, functions, classes, and for number of arguments/methods.

Line length is set to 100 for equation readability.
