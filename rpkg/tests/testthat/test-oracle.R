library("pracma")
library("gL0Learn")


P <- 10
atol <- 1e-5

theta <- matrix(stats::rnorm(P * P), P, P)
theta <- (theta + t(theta)) / 2

lows <- matrix(runif(P * P), P, P)
lows <- -(lows + t(lows)) / 2

highs <- matrix(runif(P * P), P, P)
highs <- (highs + t(highs)) / 2

l0 <- matrix(runif(P * P), P, P)
l0 <- (l0 + t(l0)) / 2

l1 <- matrix(runif(P * P), P, P)
l1 <- (l1 + t(l1)) / 2

l2 <- matrix(runif(P * P), P, P)
l2 <- (l2 + t(l2)) / 2

zeros <- matrix(0, P, P)

penalty_L0L1L2 <- new(WrappedPenaltyL0L1L2_mat, l0, l1, l2)
penalty_L0L2 <- new(WrappedPenaltyL0L2_mat, l0, l2)
penalty_L0 <- new(WrappedPenaltyL0_mat, l0)
penalty_L0ZeroL1L2 <- new(WrappedPenaltyL0L1L2_mat, l0, zeros, l2)
penalty_L0ZeroL1ZeroL2 <- new(WrappedPenaltyL0L1L2_mat, l0, zeros, zeros)
penalty_L0ZeroL2 <- new(WrappedPenaltyL0L2_mat, l0, zeros)
bounds <- new(Bounds_mat, lows, highs)
noBounds <- new(NoBounds)


test_that("Oracle matches graphed solution for L0 with bounds", {
  theta_opt <- matrix(NA, P, P)
  for (i in 1:P) {
    for (j in 1:P) {
      theta_opt[i, j] <- gL0Learn.linear_search(theta[i, j],
        l0 = l0[i, j],
        lows = lows[i, j],
        highs = highs[i, j],
        atol = atol
      )
    }
  }

  results_L0 <- penalty_L0$prox_mat_mat(theta, bounds)

  expect_equal(theta_opt, results_L0, tolerance = atol)
})

test_that("Oracle matches graphed solution for L0 without bounds", {
  theta_opt <- matrix(NA, P, P)
  for (i in 1:P) {
    for (j in 1:P) {
      theta_opt[i, j] <- gL0Learn.linear_search(theta[i, j],
        l0 = l0[i, j],
        lows = min(-0.1, 1.1 * theta[i, j]),
        highs = max(+0.1, 1.1 * theta[i, j]),
        atol = atol
      )
    }
  }

  results_L0 <- penalty_L0$prox_mat_(theta, noBounds)

  expect_equal(theta_opt, results_L0, tolerance = atol)
})

test_that("Oracle matches graphed solution for L0L2 with bounds", {
  theta_opt <- matrix(NA, P, P)
  for (i in 1:P) {
    for (j in 1:P) {
      theta_opt[i, j] <- gL0Learn.linear_search(theta[i, j],
        l0 = l0[i, j],
        l2 = l2[i, j],
        lows = lows[i, j],
        highs = highs[i, j],
        atol = atol
      )
    }
  }

  results_L0L2 <- penalty_L0L2$prox_mat_mat(theta, bounds)

  expect_equal(theta_opt, results_L0L2, tolerance = atol)
})

test_that("Oracle matches graphed solution for L0L2 without bounds", {
  theta_opt <- matrix(NA, P, P)
  for (i in 1:P) {
    for (j in 1:P) {
      theta_opt[i, j] <- gL0Learn.linear_search(theta[i, j],
        l0 = l0[i, j],
        l2 = l2[i, j],
        lows = min(-0.1, 1.1 * theta[i, j]),
        highs = max(+0.1, 1.1 * theta[i, j]),
        atol = atol
      )
    }
  }

  results_L0L2 <- penalty_L0L2$prox_mat_(theta, noBounds)

  expect_equal(theta_opt, results_L0L2, tolerance = atol)
})

test_that("Oracle matches graphed solution for L0L1L2 with bounds", {
  theta_opt <- matrix(NA, P, P)
  for (i in 1:P) {
    for (j in 1:P) {
      theta_opt[i, j] <- gL0Learn.linear_search(theta[i, j],
        l0 = l0[i, j],
        l1 = l1[i, j],
        l2 = l2[i, j],
        lows = lows[i, j],
        highs = highs[i, j],
        atol = atol
      )
    }
  }

  results_L0L1L2 <- penalty_L0L1L2$prox_mat_mat(theta, bounds)

  expect_equal(theta_opt, results_L0L1L2, tolerance = atol)
})

test_that("Oracle matches graphed solution for L0L1L2 without bounds", {
  theta_opt <- matrix(NA, P, P)
  for (i in 1:P) {
    for (j in 1:P) {
      theta_opt[i, j] <- gL0Learn.linear_search(theta[i, j],
        l0 = l0[i, j],
        l1 = l1[i, j],
        l2 = l2[i, j],
        lows = min(-0.1, 1.1 * theta[i, j]),
        highs = max(+0.1, 1.1 * theta[i, j]),
        atol = atol
      )
    }
  }

  results_L0L1L2 <- penalty_L0L1L2$prox_mat_(theta, noBounds)

  expect_equal(theta_opt, results_L0L1L2, tolerance = atol)
})


test_that("Oracle L0L1L2 is consistent with L0L2 when L1 is 0 with bounds", {
  results_L0ZeroL1L2 <- penalty_L0ZeroL1L2$prox_mat_mat(theta, bounds)
  results_L0L2 <- penalty_L0L2$prox_mat_mat(theta, bounds)
  expect_equal(results_L0ZeroL1L2, results_L0L2)
})

test_that("Oracle L0L1L2 is consistent with L0L2 when L1 is 0 without bounds", {
  results_L0ZeroL1L2 <- penalty_L0ZeroL1L2$prox_mat_(theta, noBounds)
  results_L0L2 <- penalty_L0L2$prox_mat_(theta, noBounds)
  expect_equal(results_L0ZeroL1L2, results_L0L2)
})

test_that("Oracle L0L1L2, L0L2 is consistent with L0 when L1 and L2 are 0 with bounds", {
  results_L0ZeroL1ZeroL2 <- penalty_L0ZeroL1ZeroL2$prox_mat_mat(theta, bounds)
  results_L0ZeroL2 <- penalty_L0ZeroL2$prox_mat_mat(theta, bounds)
  results_L0 <- penalty_L0$prox_mat_mat(theta, bounds)

  expect_equal(results_L0ZeroL1ZeroL2, results_L0)
  expect_equal(results_L0ZeroL2, results_L0)
})
test_that("Oracle L0L1L2, L0L2 is consistent with L0 when L1 and L2 are 0 without bounds", {
  results_L0ZeroL1ZeroL2 <- penalty_L0ZeroL1ZeroL2$prox_mat_(theta, noBounds)
  results_L0ZeroL2 <- penalty_L0ZeroL2$prox_mat_(theta, noBounds)
  results_L0 <- penalty_L0$prox_mat_(theta, noBounds)

  expect_equal(results_L0ZeroL1ZeroL2, results_L0)
  expect_equal(results_L0ZeroL2, results_L0)
})
