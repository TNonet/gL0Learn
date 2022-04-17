library("gL0Learn")


test_that("Coordinate to vector to coorindate works as expected", {
  x <- matrix(1:10, 5, 2)
  expect_equal(x, test_coordinate_matrix_to_vector_to_matrix(x))

  x <- matrix(0, 0, 2)
  expect_equal(x, test_coordinate_matrix_to_vector_to_matrix(x))

  x <- matrix(1:9, 3, 3)
  expect_equal(matrix(1:6, 3, 2), test_coordinate_matrix_to_vector_to_matrix(x))
})

test_that("Union of union_of_correlated_features2 works as expected", {
  for (i in 1:10) {
    N <- as.integer(runif(1, min = 5, 20))
    P <- N - 2
    x <- matrix(rnorm(N * P), N, P)
    xtx <- t(x) %*% x
    xtx[lower.tri(xtx, diag = TRUE)] <- 0

    a <- min(abs(xtx[upper.tri(xtx, diag = FALSE)]))
    b <- max(abs(xtx[upper.tri(xtx, diag = FALSE)]))

    t <- (a + b) / 2

    xtx_gt_t <- (which(abs(xtx) > t, arr.ind = TRUE) - 1)

    g_xtx_gt_t <- test_union_of_correlated_features2(x, t)

    g_indices <- g_xtx_gt_t[, 1] * P + g_xtx_gt_t[, 2]
    R_indices <- xtx_gt_t[, 1] * P + xtx_gt_t[, 2]

    if (length(g_indices) <= 1) {
      next
    }

    expect_equal(g_xtx_gt_t[, 1] * P + g_xtx_gt_t[, 2], sort(g_indices))
    expect_equal(g_indices, sort(R_indices))
  }
})

test_that("upper_triangluar_coords works as expected", {
  expect_equal(length(upper_triangluar_coords(0)), 0)
  expect_equal(upper_triangluar_coords(2), matrix(c(0, 1), 1, 2))
  expect_equal(upper_triangluar_coords(3), matrix(c(0, 0, 1, 1, 2, 2), 3, 2))
})

test_that("check_coordinate_matrix works as expected", {
  for (i in 1:10) {
    expect_true(check_coordinate_matrix_is_valid(upper_triangluar_coords(i)))
  }

  x <- matrix(c(1, 0), 1, 2)
  expect_false(check_coordinate_matrix_is_valid(x))
  expect_false(check_coordinate_matrix_is_valid(x, for_order = FALSE))
  expect_true(check_coordinate_matrix_is_valid(x, for_upper_triangle = FALSE))
  expect_true(check_coordinate_matrix_is_valid(x, for_upper_triangle = FALSE, for_order = FALSE))

  x <- matrix(c(0, 0, 2, 1), 2, 2)
  expect_false(check_coordinate_matrix_is_valid(x))
  expect_true(check_coordinate_matrix_is_valid(x, for_order = FALSE))
  expect_false(check_coordinate_matrix_is_valid(x, for_upper_triangle = FALSE))
  expect_true(check_coordinate_matrix_is_valid(x, for_upper_triangle = FALSE, for_order = FALSE))
})

test_that("check_is_valid_coordinate_subset works as expected", {
  for (i in 1:10) {
    expect_true(check_is_valid_coordinate_subset(
      upper_triangluar_coords(i + 1),
      upper_triangluar_coords(i)
    ))
    expect_true(check_is_valid_coordinate_subset(
      upper_triangluar_coords(i),
      upper_triangluar_coords(i)
    ))
  }

  for (i in 3:10) {
    expect_false(check_is_valid_coordinate_subset(
      upper_triangluar_coords(i - 1),
      upper_triangluar_coords(i)
    ))
  }

  x1 <- matrix(c(0, 0, 0, 1, 2, 3), 3, 2)
  x2 <- matrix(c(0, 0, 1, 2), 2, 2)
  expect_true(check_is_valid_coordinate_subset(x1, x2))
  expect_true(check_is_valid_coordinate_subset(x1, x1))
  expect_true(check_is_valid_coordinate_subset(x2, x2))
  expect_false(check_is_valid_coordinate_subset(x2, x1))
})
