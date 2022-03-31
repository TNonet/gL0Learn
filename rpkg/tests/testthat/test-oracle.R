library("gL0Learn")
library("pracma")

test_that("Oracle is consistent for different types of data ", {
  P <- 10
  atol <- 1e-5

  theta <- matrix(rnorm(P * P), P, P)
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

  for (f in c("test_oracle_L0", "test_oracle_L0L2", "test_oracle_L0L1L2")) {
    results <- do.call(f, list(theta, l0, l1, l2, lows, highs))

    theta_opt <- matrix(0, P, P)

    if (!grepl("L0", f, fixed = TRUE)) {
      l0 <- matrix(0, P, P)
    }

    if (!grepl("L1", f, fixed = TRUE)) {
      l1 <- matrix(0, P, P)
    }

    if (!grepl("L2", f, fixed = TRUE)) {
      l2 <- matrix(0, P, P)
    }

    for (i in 1:P) {
      for (j in 1:P) {
        theta_opt[i, j] <- gL0Learn::gL0Learn.linear_search(theta[i, j],
          l0[i, j],
          l1[i, j],
          l2[i, j],
          lows[i, j],
          highs[i, j],
          atol = atol
        )
      }
    }

    for (name in names(results$with_bounds)) {
      expect_equal(matrix(results$with_bounds[[name]], P, P), theta_opt, info = name, tolerance = atol)
    }

    theta_opt <- matrix(0, P, P)

    for (i in 1:P) {
      for (j in 1:P) {
        theta_opt[i, j] <- gL0Learn::gL0Learn.linear_search(theta[i, j],
          l0[i, j],
          l1[i, j],
          l2[i, j],
          min(-0.1, 1.1 * theta[i, j]),
          max(+0.1, 1.1 * theta[i, j]),
          atol = atol
        )
      }
    }

    for (name in names(results$without_bounds)) {
      expect_equal(matrix(results$without_bounds[[name]], P, P), theta_opt, info = name, tolerance = atol)
    }
  }
})


test_that("Oracle L0L1L2 is consistent with L0L2 when L1 is 0 ", {
  P <- 20
  atol <- 1e-5
  theta <- matrix(rnorm(P * P), P, P)
  theta <- (theta + t(theta)) / 2

  lows <- matrix(runif(P * P), P, P)
  lows <- -(lows + t(lows)) / 2

  highs <- matrix(runif(P * P), P, P)
  highs <- (highs + t(highs)) / 2

  l0 <- matrix(runif(P * P), P, P)
  l0 <- (l0 + t(l0)) / 2

  l1 <- matrix(0, P, P)

  l2 <- matrix(runif(P * P), P, P)
  l2 <- (l2 + t(l2)) / 2

  results_L0L1L2 <- test_oracle_L0L1L2(theta, l0, l1, l2, lows, highs)
  results_L0L2 <- test_oracle_L0L2(theta, l0, l1, l2, lows, highs)

  expect_equal(results_L0L1L2, results_L0L2)
})

test_that("Oracle L0L1L2, L0L2 is consistent with L0 when L1 and L2 are 0 ", {
  P <- 10
  atol <- 1e-5
  theta <- matrix(rnorm(P * P), P, P)
  theta <- (theta + t(theta)) / 2

  lows <- matrix(runif(P * P), P, P)
  lows <- -(lows + t(lows)) / 2

  highs <- matrix(runif(P * P), P, P)
  highs <- (highs + t(highs)) / 2

  l0 <- matrix(runif(P * P), P, P)
  l0 <- (l0 + t(l0)) / 2

  l1 <- matrix(0, P, P)

  l2 <- matrix(0, P, P)

  results_L0L1L2 <- test_oracle_L0L1L2(theta, l0, l1, l2, lows, highs)
  results_L0L2 <- test_oracle_L0L2(theta, l0, l1, l2, lows, highs)
  results_L0 <- test_oracle_L0(theta, l0, l1, l2, lows, highs)

  expect_equal(results_L0L1L2, results_L0)
  expect_equal(results_L0L2, results_L0)
})
