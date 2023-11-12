# PowerFlow
TET4205 Group Project - Power Load Flow

Files:
- classes.py
- functions.py
- main.py

Solution Implementations:
- Newton-Raphson Load Flow (NR)
  - Including reactive power limits
  - Disregarding reactive power limits
  - flat start vs. DCPF output start
- Decoupled Load Flow (DLF)
  - Disregarding reactive power limits
  - flat start vs. DCPF output start
- Fast Decoupled Load Flow (FDLF)
  - Including reactive power limits - update initial estimates at end of first iteration
  - Including reactive power limits - update partial initial estimates half-way through the algorithm
  - Disregarding reactive power limits
  - flat start vs. DCPF output start
- DC Power Flow (DCPF)


Instructions for Use:
Instructions for using each method are included in the main file, separated by method. The excel file included contains test system 2. The first sheet contains the bus information and the second the line information. Instructions for changing the excel are as follows. For unknown values, leave the spot blank, do not fill in with a zero.

Sheet Initial:
Column Description
- bus_num: fill in the bus number
- bus_type: fill in 'pv', 'pq', 'slack', or leave empty
- V: fill in voltage magnitude values if not using a flat start
- T: fill in voltage angle values if not using a flat start
- P: fill in initial P values per bus
- Q: fill in initial Q values per bus
- q_lim: fill in reactive power limit on bus of choice

Sheet line_imp:
Column Description
- line: fill in line in form 'xy' where the line connects bus x to bus y
- R: fill in real part of line impedance
- X: fill in imaginary part of line impedance
- shunt_r: fill in real part of shunt
- shunt_x: fill in imaginary part of shunt
- t_x: fill in imaginary part of transformer
- t_a: fill in transformer A value


