library("gL0Learn")

penalties <- c("L0", "L0L2", "L0L1L2") # "L1", "L2", "L0L1", "L1L2" Not supported ATM
for (penalty in penalties) {
  num_penalties <- as.integer(nchar(penalty) / 2)
  test_that(paste(penalty, "works for doubles"), {
    penalty_load <- paste("new(WrappedPenalty",
      penalty,
      "_double, ",
      paste(as.list(rep(1, num_penalties)), collapse = ", "),
      ")",
      sep = ""
    )
    object <- eval(parse(text = penalty_load))

    expect_true(object$validate())
  })

  test_that(paste(penalty, "works for matricies"), {
    penalty_load <- paste("new(WrappedPenalty",
      penalty,
      "_mat, ",
      paste(as.list(rep("matrix(c(1,1,1,1), 2, 2)", num_penalties)), collapse = ", "),
      ")",
      sep = ""
    )
    object <- eval(parse(text = penalty_load))

    expect_true(object$validate())
  })
}

test_that("NoBounds works", {
  object <- new(NoBounds)
  expect_true(object$validate())
})

test_that("Bounds works for doubles", {
  object <- new(Bounds_double, -1, 1)
  expect_true(object$validate())
})

test_that("Bounds works for mats", {
  object <- new(Bounds_mat, -1 * matrix(c(1, 1, 1, 1), 2, 2), matrix(c(1, 1, 1, 1), 2, 2))
  expect_true(object$validate())
})


p <- 2
data <- gL0Learn.generate_synthetic(p, p, normalize = "covariance", model = "KR1", rho = 0.1)

bounds <- list(
  list(lows = -Inf, highs = Inf),
  list(lows = -1, highs = 1),
  list(lows = -1 * matrix(1, p, p), highs = matrix(1, p, p))
)

penalties <- list()
penalty_values <- list(1, matrix(1, p, p))
i <- 1
for (a in penalty_values) {
  penalties[[i]] <- list(l0 = 0, l1 = 0, l2 = 0)
  i <- i + 1 # L0 Penalty
  penalties[[i]] <- list(l0 = a, l1 = 0, l2 = 0)
  i <- i + 1 # L0 Penalty
  penalties[[i]] <- list(l0 = a, l1 = 0, l2 = a)
  i <- i + 1 # L0L2 Penalty
  penalties[[i]] <- list(l0 = a, l1 = a, l2 = a)
  i <- i + 1 # L0L1L2 Penalty
}

for (penalty_list in penalties) {
  for (bound_list in bounds) {
    test_that(toString(paste(
      "gL0Learn bindings work for penalty ",
      format(penalty_list), " bounds ",
      format(bound_list)
    )), {
      fitmodel <- gL0Learn.gfit(data$X,
        l0 = penalty_list$l0,
        l1 = penalty_list$l1,
        l2 = penalty_list$l2,
        lows = bound_list$lows,
        highs = bound_list$highs
      )

      expect_true(length(fitmodel$costs) > 1)
    })
  }
}
