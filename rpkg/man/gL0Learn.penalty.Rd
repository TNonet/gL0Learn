% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/oracle.R
\name{gL0Learn.penalty}
\alias{gL0Learn.penalty}
\title{gL0Learn.penalty}
\usage{
gL0Learn.penalty(l0 = 0, l1 = 0, l2 = 0)
}
\arguments{
\item{l0}{See `gL0Learn.fit` l0 penalty documentation.}

\item{l1}{See `gL0Learn.fit` l0 penalty documentation}

\item{l2}{See `gL0Learn.fit` l0 penalty documentation}
}
\description{
Creates a C++ penalty object. **Warning** This function does not
perform value checking and assumes all values conform to `gL0Learn.fit`'s 
checks. If this is not true, bare C++ errors or segfaults may occur!
The values must satisfy all of the following:
   1. `l0`, `l1`, `l2` are of the same type or are 0
   2. Type must be real scalar or real matrix
}
